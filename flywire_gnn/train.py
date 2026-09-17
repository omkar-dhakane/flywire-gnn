"""Train a baseline model on FlyWire FAFB node classification."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import FlyWireFAFB
from .models import GCN, GraphSAGE, MLP

MODELS = {"mlp": MLP, "gcn": GCN, "sage": GraphSAGE}


def evaluate(model, data, mask, model_name):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        y = data.y[mask]
    acc = (pred == y).float().mean().item()
    correct = pred == y
    f1_sum, f1_n = 0.0, 0
    for c in range(out.shape[1]):
        c_pred = pred == c
        c_true = y == c
        tp = (c_pred & c_true & (y >= 0)).sum().item()
        fp = (c_pred & ~c_true & (y >= 0)).sum().item()
        fn = (~c_pred & c_true).sum().item()
        if c_true.sum().item() == 0:
            continue
        f1_sum += 2 * tp / max(2 * tp + fp + fn, 1)
        f1_n += 1
    return acc, f1_sum / max(f1_n, 1)


def train_one(
    model_name,
    data,
    masks,
    hidden=128,
    epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    seed=42,
    device="cpu",
    log_every=25,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    data = data.to(device)
    train_mask, val_mask, test_mask = [m.to(device) for m in masks]

    model = MODELS[model_name](
        data.x.shape[1], hidden=hidden, out_dim=int(data.y.max().item()) + 1
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == epochs:
            acc_v, _ = evaluate(model, data, val_mask, model_name)
            improved = acc_v > best_val
            if improved:
                best_val = acc_v
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            if log_every and epoch % log_every == 0:
                print(
                    f"  epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  "
                    f"val_acc={acc_v:.4f}  best_val={best_val:.4f}",
                    flush=True,
                )
    model.load_state_dict(best_state)
    acc_t, f1_t = evaluate(model, data, test_mask, model_name)
    return {
        "model": model_name,
        "test_acc": round(acc_t, 4),
        "test_macro_f1": round(f1_t, 4),
        "best_val_acc": round(best_val, 4),
        "best_epoch": best_epoch,
        "epochs": epochs,
        "seed": seed,
        "min_synapses": None,
        "wall_time_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="FlyWire FAFB baseline training")
    ap.add_argument("--model", choices=list(MODELS), default="gcn")
    ap.add_argument("--models", default=None,
                    help="comma-separated list to run several, e.g. mlp,gcn,sage")
    ap.add_argument("--labels", default="super_class",
                    choices=["super_class", "cell_class", "cell_sub_class", "cell_type"])
    ap.add_argument("--min-synapses", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=None, help="dataset cache dir")
    args = ap.parse_args()

    names = args.models.split(",") if args.models else [args.model]

    print(f"Loading FlyWire FAFB v783 (labels={args.labels}, min_synapses={args.min_synapses})...")
    ds = FlyWireFAFB(root=args.root, labels=args.labels, min_synapses=args.min_synapses)
    data = ds.data
    masks = ds.splits(seed=args.seed)
    print(ds.summary())
    n_train = int(masks[0].sum())
    n_val = int(masks[1].sum())
    n_test = int(masks[2].sum())
    print(f"  splits: train={n_train:,} val={n_val:,} test={n_test:,}")

    results = []
    for name in names:
        print(f"\n=== {name.upper()} (hidden={args.hidden}, epochs={args.epochs}, seed={args.seed}) ===")
        r = train_one(
            name, data, masks, hidden=args.hidden, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
            device=args.device, log_every=25,
        )
        r["min_synapses"] = args.min_synapses
        results.append(r)
        print(f"RESULT {json.dumps(r)}")

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['model']:>5}  acc={r['test_acc']:.4f}  macro_f1={r['test_macro_f1']:.4f}  "
              f"val={r['best_val_acc']:.4f}  epochs_used={r['best_epoch']}")


if __name__ == "__main__":
    main()
