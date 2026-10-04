"""
train.py —— 只用"观测"训练

监督信号只有 a_obs：由位置序列做二阶差分得到的加速度。真实加速度 a_true 只在
打印验证指标时使用，不参与训练。

用法:  python train.py newton      (或 pairwise / mlp)
"""
import argparse, json, math, os, time
import numpy as np
import torch
from model import build

KEYS = ("x", "m", "K", "L", "mask", "a_obs", "a_true")


def load(path):
    d = np.load(path)
    return {k: torch.from_numpy(d[k]) for k in KEYS}


def rel_loss(a_pred, a_ref, mask, sigma=0.3):
    """相对误差损失：|Δa|² / (|a|² + σ²)。让弱力场（远距离）和强力场（近距离）的样本同样重要。"""
    err2 = ((a_pred - a_ref) ** 2).sum(-1)
    w = 1.0 / ((a_ref**2).sum(-1) + sigma**2)
    return (err2 * w * mask).sum() / mask.sum()


@torch.no_grad()
def evaluate(model, data, bs=8192):
    """对照真实加速度的两个指标：
    rel_rms = sqrt(Σ|Δa|² / Σ|a_true|²)，median = 每个物体相对误差 |Δa|/|a_true| 的中位数。"""
    model.eval()
    num = den = 0.0
    rel = []
    for i in range(0, len(data["m"]), bs):
        b = {k: v[i:i + bs] for k, v in data.items()}
        a = model(b["x"], b["m"], b["K"], b["L"], b["mask"])
        e = (a - b["a_true"]).norm(dim=-1)[b["mask"]]
        t = b["a_true"].norm(dim=-1)[b["mask"]]
        num += (e**2).sum().item(); den += (t**2).sum().item()
        rel.append(e / t.clamp_min(1e-9))
    model.train()
    return {"rel_rms": math.sqrt(num / den), "median": torch.cat(rel).median().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["newton", "pairwise", "mlp"])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, val = load("data/train.npz"), load("data/val.npz")
    S = len(train["m"])
    model = build(args.name)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.name}] 参数量 {n_par}，训练样本 {S}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.01)
    log, t0, run = [], time.time(), 0.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, S, (args.batch,))
        b = {k: v[idx] for k, v in train.items()}
        loss = rel_loss(model(b["x"], b["m"], b["K"], b["L"], b["mask"]), b["a_obs"], b["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 2500 == 0 or step == args.steps:
            m = evaluate(model, val)
            log.append({"step": step, "train_loss": run, **m, "time": time.time() - t0})
            print(f"[{args.name}] step {step:6d}  loss {run:.2e}  "
                  f"验证: 相对RMS误差 {m['rel_rms']:.3%}  中位相对误差 {m['median']:.3%}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    out = f"checkpoints/{args.name}{args.tag}"
    torch.save(model.state_dict(), out + ".pt")
    json.dump({"args": vars(args), "n_params": n_par, "log": log}, open(out + ".json", "w"), indent=1)
    print(f"[{args.name}] 已保存 {out}.pt", flush=True)


if __name__ == "__main__":
    main()
