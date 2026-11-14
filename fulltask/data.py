"""Shared official split and frozen-patch cache; no held-out or future policy inputs."""
import json, pickle, hashlib, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .wa_config import HISTORY_OFFSETS, FUTURE_OFFSETS, ACTION_BLOCK, PLAN_BLOCKS
ROOT = Path(os.environ.get('PUSHT_DATA_ROOT', 'data/fulltask/pusht_dataset'))
DINO_REPO = os.environ.get('DINO_REPO', 'third_party/dinov2')
DINO_WEIGHTS = os.environ.get('DINO_CHECKPOINT', 'pretrained/dinov2_vits14_pretrain.pth')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda : f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def metadata():
    lengths = pickle.load(open(ROOT / 'seq_lengths.pkl', 'rb'))
    actions = torch.load(ROOT / 'abs_actions.pth', weights_only=False).float()
    states = torch.load(ROOT / 'states.pth', weights_only=False).float()
    ids = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(42)).tolist()
    return (lengths, actions, states, ids[:185], ids[185:])

def proprio(states):
    xy = states[..., :2]
    return torch.cat([xy, torch.cat([torch.zeros_like(xy[:1]), xy[1:] - xy[:-1]])], -1)

class CachedEpisodes(Dataset):

    def __init__(self, cache, split):
        self.cache = Path(cache)
        self.manifest = json.loads((self.cache / 'manifest.json').read_text())
        self.ids = self.manifest[split]
        (self.lengths, self.actions, self.states, _, _) = metadata()
        self.maps = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['maps'] = {}
        return state

    def __len__(self):
        return len(self.ids)

    def get_seq_length(self, i):
        return int(self.lengths[self.ids[i]])

    def features(self, ep):
        if ep not in self.maps:
            self.maps[ep] = np.load(self.cache / f'episode_{ep:03d}.npy', mmap_mode='r')
        return self.maps[ep]

    def __getitem__(self, i):
        ep = self.ids[i]
        n = self.get_seq_length(i)
        return (torch.from_numpy(self.features(ep)[:, -1]), self.actions[ep, :n], torch.zeros(n, 1))

    def get_frames(self, i, frames):
        return tuple((v[frames] for v in self[i]))

class WAWindows(Dataset):

    def __init__(self, cache, split):
        self.episodes = CachedEpisodes(cache, split)
        self.index = [(ep, t) for ep in self.episodes.ids for t in range(int(self.episodes.lengths[ep]))]
        self.goal = torch.from_numpy(np.load(Path(cache) / 'goal.npy'))
        self.props = {ep: proprio(self.episodes.states[ep, :int(self.episodes.lengths[ep])]) for ep in self.episodes.ids}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        (ep, t) = self.index[i]
        n = int(self.episodes.lengths[ep])
        features = self.episodes.features(ep)
        h = t + np.asarray(HISTORY_OFFSETS)
        f = t + np.asarray(FUTURE_OFFSETS)
        a = t + np.arange(PLAN_BLOCKS * ACTION_BLOCK)
        return dict(history=torch.from_numpy(features[h.clip(0, n - 1)].copy()), history_valid=torch.from_numpy(h >= 0), proprio=self.props[ep][h.clip(0, n - 1)], goal=self.goal, future=torch.from_numpy(features[f.clip(0, n - 1)].copy()), future_proprio=self.props[ep][f.clip(0, n - 1)], future_mask=torch.from_numpy(f < n), actions=self.episodes.actions[ep, a.clip(0, n - 1)].reshape(PLAN_BLOCKS, ACTION_BLOCK, 2), action_mask=torch.from_numpy((a < n).reshape(PLAN_BLOCKS, ACTION_BLOCK)))

def statistics(train_ids):
    (lengths, actions, states, _, _) = metadata()
    a = torch.cat([actions[i, :int(lengths[i])] for i in train_ids])
    p = torch.cat([proprio(states[i, :int(lengths[i])]) for i in train_ids])
    return dict(action_mean=a.mean(0), action_std=a.std(0).clamp_min(0.001), proprio_mean=p.mean(0), proprio_std=p.std(0).clamp_min(0.001))

def load_dino(device):
    model = torch.hub.load(DINO_REPO, 'dinov2_vits14', source='local', pretrained=False)
    model.load_state_dict(torch.load(DINO_WEIGHTS, map_location='cpu', weights_only=True))
    return model.to(device).eval().requires_grad_(False)

@torch.no_grad()
def encode(dino, pixels):
    if pixels.shape[-1] == 3:
        pixels = pixels.permute(0, 3, 1, 2).float() / 255
    pixels = pixels.to(next(dino.parameters()).device)
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (pixels - mean) / std
    return torch.stack(dino.get_intermediate_layers(x, n=(3, 7, 11), norm=True), 1)
