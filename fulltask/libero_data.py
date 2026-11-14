"""LIBERO Goal: visual-only dual-view H5/A5 data with training-only task goals."""
import json, os
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import DINO_REPO,DINO_WEIGHTS,sha,load_dino
from .vision import encode
ROOT=Path(os.environ.get('LIBERO_DATA_ROOT','data/fulltask/libero_dataset/libero_goal'))
VIEWS=('agentview_image','robot0_eye_in_hand_image')

@lru_cache(None)
def metadata():
    tasks=sorted(p for p in ROOT.iterdir() if p.is_dir());assert len(tasks)==10
    demos=[];train=[];val=[];goals=[]
    for task_id,task in enumerate(tasks):
        paths=sorted((p for p in task.iterdir() if p.is_dir()),key=lambda p:int(p.name.split('_')[-1]));assert len(paths)==50
        order=torch.randperm(50,generator=torch.Generator().manual_seed(42+task_id)).tolist()
        # Fixed goal template must belong to the training split.
        start=len(demos);goals.append(start+order[0]);train.extend(start+i for i in order[:45]);val.extend(start+i for i in order[45:])
        for path in paths:
            a=np.load(path/'actions.npy').astype('float32');assert a.ndim==2 and a.shape[1]==7 and len(a)>5 and np.isfinite(a).all()
            demos.append(dict(path=str(path),task_id=task_id,length=len(a)))
    return demos,train,val,goals,[p.name for p in tasks]

def actions(ep):return torch.from_numpy(np.load(Path(metadata()[0][ep]['path'])/'actions.npy').astype('float32'))

def pixels(ep,view):
    p=Path(metadata()[0][ep]['path'])/VIEWS[view]
    if p.with_suffix('.pth').exists():
        x=torch.load(p.with_suffix('.pth'),map_location='cpu',weights_only=False)
        assert x.shape[1:]==(224,224,3);return x
    import av
    with av.open(str(p.with_suffix('.mp4'))) as video:
        frames=[f.to_ndarray(format='rgb24') for f in video.decode(video=0)]
    x=torch.from_numpy(np.stack(frames));assert x.shape[1:]==(224,224,3);return x

class WAWindows(Dataset):
    def __init__(self,cache,split):
        self.cache=Path(cache);self.manifest=json.loads((self.cache/'manifest.json').read_text());self.ids=self.manifest[split]
        self.demos=metadata()[0];self.acts={ep:actions(ep) for ep in self.ids};self.maps={}
        self.goals=torch.from_numpy(np.load(self.cache/'goals.npy'))
        self.index=[(ep,t) for ep in self.ids for t in range(self.demos[ep]['length'])]
    def __getstate__(self):
        state=self.__dict__.copy();state['maps']={};return state
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        ep,t=self.index[i];n=self.demos[ep]['length']
        if ep not in self.maps:self.maps[ep]=np.load(self.cache/f'episode_{ep:05d}.npy',mmap_mode='r')
        f=self.maps[ep];h=t+np.arange(-4,1);a=t+np.arange(5);future=np.asarray([t+5])
        return dict(history=torch.from_numpy(f[h.clip(0,n-1)].copy()),history_valid=torch.from_numpy(h>=0),
                    goal=self.goals[self.demos[ep]['task_id']],actions=self.acts[ep][a.clip(0,n-1)][None],
                    future=torch.from_numpy(f[future.clip(0,n-1)].copy()),future_mask=torch.from_numpy(future<n))

def statistics(ids):
    a=torch.cat([actions(i) for i in ids]);return dict(action_mean=a.mean(0),action_std=a.std(0).clamp_min(1e-3))
