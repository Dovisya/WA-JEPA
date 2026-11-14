"""Training-only shared 5x7 residual-VQ codebook, separate from policy training."""
import argparse, json
from pathlib import Path
import torch
from .libero_data import metadata, actions, sha
from .libero_model import make_vq

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cache = Path(args.cache)
    m = json.loads((cache / 'manifest.json').read_text())
    passes = 1000
    signature = dict(passes=passes, cache_sha256=sha(cache / 'manifest.json'), train=m['train'])
    if (cache / 'codebook.pt').exists():
        assert torch.load(cache / 'codebook.pt', weights_only=False)['signature'] == signature
        print('CODEBOOK_ALREADY_READY')
        return
    (demos, _, _, _, _) = metadata()
    chunks = []
    for ep in m['train']:
        n = demos[ep]['length']
        a = actions(ep)
        for t in range(n):
            start = max(0, t - 4)
            seq = a[start:min(t + 5, n)]
            if t < 4:
                seq = torch.cat([seq[:1].expand(4 - t, -1), seq])
            if len(seq) < 9:
                seq = torch.cat([seq, seq[-1:].expand(9 - len(seq), -1)])
            chunks.extend((seq[k:k + 5] for k in range(5)))
    chunks = torch.unique(torch.stack(chunks).flatten(1), dim=0).reshape(-1, 5, 7).cuda()
    vq = make_vq().cuda().train()
    opt = torch.optim.Adam(vq.parameters(), lr=0.001, weight_decay=0.0001)
    start = 0
    last_loss = None
    snapshot = cache / 'codebook_latest.pt'
    if snapshot.exists():
        c = torch.load(snapshot, weights_only=False)
        assert c['signature'] == signature
        vq.load_state_dict(c['model'])
        opt.load_state_dict(c['optimizer'])
        start = c['pass'] + 1
        last_loss = c.get('last_loss')
        torch.set_rng_state(c['rng_cpu'])
        torch.cuda.set_rng_state(c['rng_cuda'])
    for epoch in range(start, passes):
        for idx in torch.randperm(len(chunks), device='cuda').split(1024):
            (loss, _, _) = vq(chunks[idx])
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        last_loss = float(loss)
        if epoch % 10 == 0 or epoch == passes - 1:
            print('VQ_PASS', epoch + 1, 'loss', float(loss), 'unique_chunks', len(chunks), flush=True)
            state = dict(model=vq.state_dict(), optimizer=opt.state_dict(), signature=signature, pass_=epoch, last_loss=last_loss, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())
            state['pass'] = state.pop('pass_')
            torch.save(state, str(snapshot) + '.tmp')
            Path(str(snapshot) + '.tmp').replace(snapshot)
    final = dict(model=vq.state_dict(), signature=signature, unique_chunks=len(chunks), loss=last_loss)
    torch.save(final, cache / 'codebook.tmp.pt')
    (cache / 'codebook.tmp.pt').replace(cache / 'codebook.pt')
    print('CODEBOOK_PASS', flush=True)
if __name__ == '__main__':
    main()
