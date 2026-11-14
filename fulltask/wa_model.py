"""WA-JEPA full-task VQ-BeT adaptation."""
import torch
from torch import nn
from torch.nn import functional as F
from models.vq_behavior_transformer.vqvae import VqVae
from models.vq_behavior_transformer.utils import MLP
from models.vq_behavior_transformer.bet import FocalLoss
from .architecture import WorldActionJEPA, WorldActionPlanner
from .data import DINO_REPO, DINO_WEIGHTS
from .wa_config import HISTORY_OFFSETS, PLAN_BLOCKS, ACTION_BLOCK


def patch_policy_action_loss(raw, logits, target, codes):
    """Native VQ-BeT criterion in pixel coordinates, including padded tails."""
    bc = F.l1_loss(raw, target)
    flat_logits = logits.reshape(-1, 2, 16)
    flat_codes = codes.reshape(-1, 2)
    criterion = FocalLoss(gamma=2.0)
    cls = (5.0 * criterion(flat_logits[:, 0], flat_codes[:, 0])
           + 0.5 * criterion(flat_logits[:, 1], flat_codes[:, 1]))
    return 10.0 * bc + cls, cls, bc

def make_vq():
    return VqVae(input_dim_h=5,input_dim_w=2,n_latent_dims=512,vqvae_n_embed=16,
                 vqvae_groups=2,act_scale=500.)

class VQActionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vq = make_vq().requires_grad_(False).eval()
        self.bins = MLP(in_channels=dim,hidden_channels=[1024,1024,32])
        self.offsets = MLP(in_channels=dim,hidden_channels=[1024,1024,320])
        self.records = []
    def train(self, mode=True):
        super().train(mode); self.vq.eval(); return self
    def forward(self,x):
        b,t,_ = x.shape
        logits = self.bins(x).reshape(b,t,2,16)
        codes = torch.multinomial(logits.softmax(-1).reshape(-1,16),1).reshape(b,t,2)
        with torch.no_grad():
            centers = self.vq.get_action_from_latent(self.vq.draw_code_forward(codes))
        offsets = self.offsets(x).reshape(b,t,2,16,5,2)
        selected = offsets.gather(3,codes[...,None,None,None].expand(-1,-1,-1,1,5,2)).squeeze(3).sum(2)
        raw = centers + selected
        self.records.append((raw,logits))
        # Explicit absolute-pixel action adapter; no tanh saturation of VQ output.
        return (raw/256.-1.).flatten(-2)

class NoTemporalProxy(nn.Module):
    def forward(self,future,goal):
        z = future.new_zeros(future.shape[:2])
        return dict(success_logits=z,progress_logits=z,goal_distance=z)

class VQPlanner(WorldActionPlanner):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.action_decoder = VQActionHead(cfg.embed_dim)
        self.evaluator = NoTemporalProxy()
    def bound_actions(self,x): return x
    def forward(self,*args):
        self.action_decoder.records = []
        return super().forward(*args)

class WAJEPAVQ(WorldActionJEPA):
    def __init__(self,stats):
        super().__init__(dino_repo=DINO_REPO,dino_checkpoint=DINO_WEIGHTS,
                         history_size=len(HISTORY_OFFSETS),horizon=PLAN_BLOCKS,
                         action_block=ACTION_BLOCK)
        self.planner = VQPlanner(self.cfg)
        del self.inverse_head
        del self.encoder.global_pool, self.encoder.global_norm, self.encoder.global_query
        self.set_action_stats(stats['action_mean'],stats['action_std'])
        self.set_proprio_stats(stats['proprio_mean'],stats['proprio_std'])
    @property
    def vq(self): return self.planner.action_decoder.vq
    def pool(self,features):
        shape=features.shape[:-3]; x=features.reshape(-1,3,256,384); e=self.encoder
        patches=sum(w*p(x[:,i]) for i,(w,p) in enumerate(zip(e.layer_logits.softmax(0),e.layer_projections)))
        scene=e.scene_queries.expand(len(x),-1,-1)
        for layer in e.scene_pool: scene=layer(scene,patches)
        return e.scene_norm(scene).reshape(*shape,8,192)
    def policy(self,batch):
        scenes=self.pool(batch['history']); goal=self.pool(batch['goal'])
        _,by_time=self.fuse_context(scenes,batch['proprio'],batch['history_valid'])
        context,goal_memory=self.match_goal(by_time,goal)
        out=self.planner(context,scenes[:,-1],goal,goal_memory)
        out['raw_actions']=self.planner.action_decoder.records[-1][0]
        return out,context,scenes[:,-1]
    def action_losses(self,batch):
        target=batch['actions']
        with torch.no_grad(): _,codes=self.vq.get_code(target)
        losses=[]; cls_losses=[]; bc_losses=[]
        for raw,logits in self.planner.action_decoder.records:
            stage_loss,cls,bc=patch_policy_action_loss(raw,logits,target,codes)
            losses.append(stage_loss); cls_losses.append(cls); bc_losses.append(bc)
        # Retain WA's intermediate-stage auxiliary supervision and world feedback.
        loss=losses[-1]+.25*torch.stack(losses[:-1]).mean()
        return loss,cls_losses[-1],bc_losses[-1]
    def forward(self,batch,bc_only=False):
        out,context,current=self.policy(batch)
        if 'actions' not in batch: return out
        action_loss,cls,bc=self.action_losses(batch)
        loss=action_loss
        wm=loss.new_zeros(()); prop=loss.new_zeros(())
        if not bc_only:
            was_training=self.encoder.training; self.encoder.eval()
            with torch.no_grad(): target=self.pool(batch['future'])
            self.encoder.train(was_training)
            teacher,teacher_prop=self.planner.world(context,current,(batch['actions']/256.-1.).flatten(-2))
            mask=batch['future_mask'].float()
            wm=((teacher-target).square().mean((-1,-2))*mask).sum()/mask.sum().clamp_min(1)
            target_prop=self.normalize_proprio(batch['future_proprio'])
            prop=(F.smooth_l1_loss(teacher_prop,target_prop,reduction='none').mean(-1)*mask).sum()/mask.sum().clamp_min(1)
            loss=loss+wm+.25*prop
        return dict(loss=loss,actions=out['raw_actions'],future=out['future_scene'],
                    metrics=dict(bc=bc.detach(),classification=cls.detach(),world=wm.detach(),proprio=prop.detach()))
