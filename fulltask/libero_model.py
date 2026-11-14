"""WA-JEPA full-task VQ-BeT adaptation."""
import torch
from torch import nn
from .architecture import WorldActionJEPA,WorldActionPlanner
from .wa_model import WAJEPAVQ,NoTemporalProxy,patch_policy_action_loss
from .libero_data import DINO_REPO,DINO_WEIGHTS
from models.vq_behavior_transformer.vqvae import VqVae
from models.vq_behavior_transformer.utils import MLP

def make_vq():
    return VqVae(input_dim_h=5,input_dim_w=7,n_latent_dims=512,vqvae_n_embed=16,vqvae_groups=2,act_scale=1.)

class ZeroOutput(nn.Module):
    def __init__(self,dim):super().__init__();self.dim=dim
    def forward(self,x):return x.new_zeros(*x.shape[:-1],self.dim)

class LiberoVQHead(nn.Module):
    def __init__(self,dim):
        super().__init__();self.vq=make_vq().requires_grad_(False).eval()
        self.bins=MLP(in_channels=dim,hidden_channels=[1024,1024,32])
        self.offsets=MLP(in_channels=dim,hidden_channels=[1024,1024,1120]);self.records=[]
    def train(self,mode=True):super().train(mode);self.vq.eval();return self
    def forward(self,x):
        b,t,_=x.shape;logits=self.bins(x).reshape(b,t,2,16)
        codes=torch.multinomial(logits.softmax(-1).reshape(-1,16),1).reshape(b,t,2)
        with torch.no_grad():centers=self.vq.get_action_from_latent(self.vq.draw_code_forward(codes))
        offsets=self.offsets(x).reshape(b,t,2,16,5,7)
        selected=offsets.gather(3,codes[...,None,None,None].expand(-1,-1,-1,1,5,7)).squeeze(3).sum(2)
        raw=centers+selected;self.records.append((raw,logits))
        # Libero controls already use [-1,1], unlike PushT absolute pixel targets.
        return raw.clamp(-1,1).flatten(-2)

class LiberoPlanner(WorldActionPlanner):
    def __init__(self,cfg):
        super().__init__(cfg);self.action_decoder=LiberoVQHead(cfg.embed_dim);self.evaluator=NoTemporalProxy()
        self.world.proprio_head=ZeroOutput(1);self.future_proprio_encoder=ZeroOutput(cfg.embed_dim)
    def bound_actions(self,x):return x
    def forward(self,*args):self.action_decoder.records=[];return super().forward(*args)

class LiberoWA(WAJEPAVQ):
    def __init__(self,stats):
        WorldActionJEPA.__init__(self,dino_repo=DINO_REPO,dino_checkpoint=DINO_WEIGHTS,history_size=5,horizon=1,action_block=5,raw_action_dim=7,proprio_dim=1)
        self.planner=LiberoPlanner(self.cfg)
        del self.inverse_head,self.encoder.global_pool,self.encoder.global_norm,self.encoder.global_query
        del self.proprio_encoder,self.proprio_type
        self.view_embedding=nn.Parameter(torch.randn(1,2,1,self.cfg.embed_dim)*.02)
        self.set_action_stats(stats['action_mean'],stats['action_std'])
    def pool(self,features):
        # (..., views=2, DINO layers=3, patches=256, width=384) -> (..., 8, 192).
        shape=features.shape[:-4];assert features.shape[-4:]==(2,3,256,384)
        x=features.reshape(-1,2,3,256,384);e=self.encoder
        patches=sum(w*p(x[:,:,i]) for i,(w,p) in enumerate(zip(e.layer_logits.softmax(0),e.layer_projections)))
        patches=(patches+self.view_embedding).flatten(1,2)
        scene=e.scene_queries.expand(len(x),-1,-1)
        for layer in e.scene_pool:scene=layer(scene,patches)
        return e.scene_norm(scene).reshape(*shape,8,192)
    def action_losses(self,batch):
        target=batch['actions']
        with torch.no_grad():_,codes=self.vq.get_code(target)
        losses=[];classes=[];bcs=[]
        for raw,logits in self.planner.action_decoder.records:
            _,cls,bc=patch_policy_action_loss(raw,logits,target,codes)
            # Official LIBERO VQ-BeT offset coefficient for normalized 7D controls.
            losses.append(100*bc+cls);classes.append(cls);bcs.append(bc)
        return losses[-1]+.25*torch.stack(losses[:-1]).mean(),classes[-1],bcs[-1]
    def fuse_visual(self,scene,valid):
        b,t,n,d=scene.shape
        x=scene+self.visual_type.view(1,1,1,-1)+self.context_time+self.history_valid_embedding(valid.long()).unsqueeze(2)
        x=x.reshape(b*t,n,d)
        for layer in self.frame_fusion:x=layer(x)
        x=x.reshape(b,t*n,d)
        for layer in self.temporal_fusion:x=layer(x)
        return self.fusion_norm(x).reshape(b,t,n,d)
    def policy(self,batch):
        scenes=self.pool(batch['history']);goal=self.pool(batch['goal'])
        context,goal_memory=self.match_goal(self.fuse_visual(scenes,batch['history_valid']),goal)
        out=self.planner(context,scenes[:,-1],goal,goal_memory)
        out['raw_actions']=self.planner.action_decoder.records[-1][0]
        return out,context,scenes[:,-1]
    def forward(self,batch,bc_only=False):
        out,context,current=self.policy(batch)
        if 'actions' not in batch:return out
        action_loss,cls,bc=self.action_losses(batch);wm=action_loss.new_zeros(())
        if not bc_only:
            was_training=self.encoder.training;self.encoder.eval()
            with torch.no_grad():target=self.pool(batch['future'])
            self.encoder.train(was_training)
            teacher,_=self.planner.world(context,current,batch['actions'].clamp(-1,1).flatten(-2))
            mask=batch['future_mask'].float()
            wm=((teacher-target).square().mean((-1,-2))*mask).sum()/mask.sum().clamp_min(1)
        return dict(loss=action_loss+wm,actions=out['raw_actions'],future=out['future_scene'],metrics=dict(bc=bc.detach(),classification=cls.detach(),world=wm.detach()))
