"""One writer fits a shared training-only 5-action codebook; workers load it."""
import argparse, json
from pathlib import Path
import torch
from .data import metadata, sha
from .wa_model import make_vq

def chunks_for_training(ids):
    (lengths, actions, _, _, _) = metadata()
    chunks = []
    for ep in ids:
        n = int(lengths[ep])
        a = actions[ep, :n]
        for t in range(n):
            start = max(0, t - 4)
            end = t + 5
            seq = a[start:min(end, n)]
            if t < 4:
                seq = torch.cat([seq[:1].expand(4 - t, -1), seq])
            if len(seq) < 9:
                seq = torch.cat([seq, seq[-1:].expand(9 - len(seq), -1)])
            chunks.extend((seq[k:k + 5] for k in range(5)))
    return torch.unique(torch.stack(chunks).flatten(1), dim=0).reshape(-1, 5, 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cache = Path(args.cache)
    manifest = json.loads((cache / 'manifest.json').read_text())
    path = cache / 'codebook.pt'
    passes = 1000
    signature = dict(passes=passes, cache_sha256=sha(cache / 'manifest.json'), train=manifest['train'])
    if path.exists():
        assert torch.load(path, weights_only=False)['signature'] == signature
        print('Codebook already fit')
        return
    chunks = chunks_for_training(manifest['train']).cuda()
    vq = make_vq().cuda().train()
    opt = torch.optim.Adam(vq.parameters(), lr=0.001, weight_decay=0.0001)
    for epoch in range(passes):
        for idx in torch.randperm(len(chunks), device='cuda').split(1024):
            (loss, _, _) = vq(chunks[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0 or epoch == passes - 1:
            print('vq_pass', epoch, 'loss', float(loss), 'unique_chunks', len(chunks), flush=True)
    torch.save(dict(model=vq.state_dict(), signature=signature, unique_chunks=len(chunks), loss=float(loss)), str(path) + '.tmp')
    Path(str(path) + '.tmp').replace(path)
if __name__ == '__main__':
    main()
