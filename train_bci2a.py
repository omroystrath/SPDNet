#!/usr/bin/env python3
"""
train_bci2a.py - Train GVSPD-Net on BCI Competition IV dataset 2a

Three model variants (--model):
  gvspd_fixed      : GV transform with frozen support (training correlation)
  gvspd_learnable  : GV transform with learnable SPD support (init from data)
  spdnet_baseline  : per-trial covariance -> SPDNet (no GV, ablation)

Two evaluation modes (--mode):
  subject_specific    : train session 1, test session 2, per subject
  subject_independent : leave-one-subject-out (LOSO)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
except ImportError:
    print("ERROR: pip install moabb")
    sys.exit(1)

try:
    import mne
    mne.set_log_level("WARNING")
except ImportError:
    print("ERROR: pip install mne")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from gvspd_net import GVSPDNet, SPDNetBaseline
from graph_variate import pearson_correlation_matrix, ensure_spd


# =============================================================
#  Globals set from args
# =============================================================
DTYPE = torch.float32
DEVICE = torch.device("cpu")


def _setup_cuda():
    """Use magma backend for linalg on GPU -- more robust than cusolver for batched ops."""
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except Exception:
            pass  # old pytorch or no magma


def to_tensor(x_np):
    """Convert numpy array to tensor with the global dtype on the global device."""
    t = torch.from_numpy(x_np.astype(np.float32 if DTYPE == torch.float32
                                      else np.float64))
    # replace any NaN/Inf from bad EEG channels
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t.to(DEVICE)


# =============================================================
#  Data loading
# =============================================================
def load_subject_sessions(subject_id, tmin, tmax):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4, fmin=4.0, fmax=38.0,
        tmin=tmin, tmax=tmax, resample=250.0,
    )
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    sessions = meta["session"].values
    unique_sessions = sorted(set(sessions))
    label_map = {lab: i for i, lab in enumerate(sorted(set(labels)))}
    y_int = np.array([label_map[l] for l in labels])

    if len(unique_sessions) >= 2:
        m1 = sessions == unique_sessions[0]
        m2 = sessions == unique_sessions[1]
    else:
        n = len(labels)
        idx = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        m1 = np.zeros(n, dtype=bool)
        m1[idx[:split]] = True
        m2 = ~m1

    return (X[m1], y_int[m1], X[m2], y_int[m2], label_map)


def load_all_subjects(subjects, tmin, tmax):
    data = {}
    for s in subjects:
        s1X, s1y, s2X, s2y, _ = load_subject_sessions(s, tmin, tmax)
        data[s] = (s1X, s1y, s2X, s2y)
        print(f"  Subject {s}: ses1 {s1X.shape}  ses2 {s2X.shape}  "
              f"T={s1X.shape[2]}")
    return data


# =============================================================
#  Support computation
# =============================================================
def compute_support_from_data(X_np):
    """Mean Pearson correlation across trials -> (C, C) SPD tensor on device."""
    X_t = to_tensor(X_np)
    C = pearson_correlation_matrix(X_t).mean(dim=0)
    return ensure_spd(C, eps=1e-3)


# =============================================================
#  Model construction
# =============================================================
def build_model(n_channels, n_classes, args):
    bimap_dims = args.bimap_dims
    if bimap_dims is None:
        bimap_dims = [max(n_channels // 2, 6), max(n_channels // 4, 4)]

    if args.model == "spdnet_baseline":
        return SPDNetBaseline(
            n_channels=n_channels, n_classes=n_classes,
            bimap_dims=bimap_dims,
        ).to(dtype=DTYPE, device=DEVICE)

    elif args.model == "gvspd_fixed":
        return GVSPDNet(
            n_channels=n_channels, n_classes=n_classes,
            bimap_dims=bimap_dims, support_mode="fixed",
            temporal_pool=args.pool,
            n_windows=args.n_windows,
        ).to(dtype=DTYPE, device=DEVICE)

    elif args.model == "gvspd_learnable":
        return GVSPDNet(
            n_channels=n_channels, n_classes=n_classes,
            bimap_dims=bimap_dims, support_mode=args.support,
            temporal_pool=args.pool,
            bimap_rank=args.support_rank,
            lr_support=args.lr_support,
            n_windows=args.n_windows,
        ).to(dtype=DTYPE, device=DEVICE)

    else:
        raise ValueError(f"Unknown model: {args.model}")


# =============================================================
#  Training / eval
# =============================================================
def train_one_epoch(model, optimiser, X, y, batch_size):
    model.train()
    n = X.shape[0]
    perm = torch.randperm(n, device=DEVICE)
    total_loss, total_correct = 0.0, 0

    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        xb, yb = X[idx], y[idx]

        logits = model(xb)
        loss = F.cross_entropy(logits.float(), yb)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(idx)
        total_correct += (logits.argmax(1) == yb).sum().item()

    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, X, y, batch_size):
    model.eval()
    n = X.shape[0]
    total_loss, total_correct = 0.0, 0

    for start in range(0, n, batch_size):
        xb, yb = X[start:start+batch_size], y[start:start+batch_size]
        logits = model(xb)
        loss = F.cross_entropy(logits.float(), yb)
        total_loss += loss.item() * len(xb)
        total_correct += (logits.argmax(1) == yb).sum().item()

    return total_loss / n, total_correct / n


def run_training(model, X_train, y_train, X_test, y_test, args, tag):
    optimiser = model.make_optimiser(lr=args.lr, lr_support=args.lr_support)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  dtype: {DTYPE}  |  device: {DEVICE}")
    print(f"  Train: {X_train.shape[0]} x T={X_train.shape[2]}  |  "
          f"Test: {X_test.shape[0]} x T={X_test.shape[2]}")

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, optimiser, X_train, y_train, args.batch_size)
        te_loss, te_acc = evaluate(model, X_test, y_test, args.batch_size)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        if te_acc > best_acc:
            best_acc = te_acc

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [{epoch:3d}/{args.epochs}]  "
                  f"train {tr_loss:.4f} / {tr_acc:.3f}  |  "
                  f"test {te_loss:.4f} / {te_acc:.3f}  ({dt:.1f}s)")

    print(f"  --> Best: {best_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out_dir}/{tag}.pt")
    if HAS_MPL:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history["train_loss"], label="train")
        ax[0].plot(history["test_loss"], label="test")
        ax[0].set_title(f"{tag} loss"); ax[0].legend()
        ax[1].plot(history["train_acc"], label="train")
        ax[1].plot(history["test_acc"], label="test")
        ax[1].set_title(f"{tag} acc"); ax[1].legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_dir}/{tag}.png", dpi=150)
        plt.close()

    return best_acc


# =============================================================
#  Subject-specific
# =============================================================
def run_subject_specific(args):
    print("\n  Mode: SUBJECT-SPECIFIC")
    data = load_all_subjects(args.subjects, args.tmin, args.tmax)
    results = {}

    for subj in args.subjects:
        print(f"\n{'='*60}\n  Subject {subj}  |  {args.model}\n{'='*60}")
        s1X, s1y, s2X, s2y = data[subj]
        n_ch, n_cls = s1X.shape[1], len(np.unique(s1y))

        # pre-load to device
        X_tr, y_tr = to_tensor(s1X), torch.from_numpy(s1y).long().to(DEVICE)
        X_te, y_te = to_tensor(s2X), torch.from_numpy(s2y).long().to(DEVICE)

        model = build_model(n_ch, n_cls, args)

        if args.model == "gvspd_fixed":
            model.set_fixed_support(compute_support_from_data(s1X))
            print(f"  Support: frozen from subject {subj} session 1")
        elif args.model == "gvspd_learnable":
            model.gv.learnable_support.reinit_from_data(to_tensor(s1X))
            model.gv._support_initialised = True
            print(f"  Support: learnable, init from subject {subj} session 1")

        try:
            acc = run_training(model, X_tr, y_tr, X_te, y_te, args,
                               f"ss_{args.model}_sub{subj}")
            results[subj] = acc
        except Exception as e:
            print(f"  x Failed: {e}")
            import traceback; traceback.print_exc()
            results[subj] = None

    return results


# =============================================================
#  Subject-independent (LOSO)
# =============================================================
def run_subject_independent(args):
    print("\n  Mode: SUBJECT-INDEPENDENT (LOSO)")
    data = load_all_subjects(args.subjects, args.tmin, args.tmax)
    results = {}

    for held_out in args.subjects:
        print(f"\n{'='*60}\n  Held-out: {held_out}  |  {args.model}\n{'='*60}")

        train_Xs, train_ys = [], []
        for s in args.subjects:
            if s == held_out:
                continue
            s1X, s1y, s2X, s2y = data[s]
            train_Xs.extend([s1X, s2X])
            train_ys.extend([s1y, s2y])

        X_tr_np = np.concatenate(train_Xs, axis=0)
        y_tr_np = np.concatenate(train_ys, axis=0)

        s1X, s1y, s2X, s2y = data[held_out]
        X_te_np = np.concatenate([s1X, s2X], axis=0)
        y_te_np = np.concatenate([s1y, s2y], axis=0)

        n_ch, n_cls = X_tr_np.shape[1], len(np.unique(y_tr_np))

        X_tr, y_tr = to_tensor(X_tr_np), torch.from_numpy(y_tr_np).long().to(DEVICE)
        X_te, y_te = to_tensor(X_te_np), torch.from_numpy(y_te_np).long().to(DEVICE)

        model = build_model(n_ch, n_cls, args)

        if args.model == "gvspd_fixed":
            model.set_fixed_support(compute_support_from_data(X_tr_np))
            print(f"  Support: frozen global ({X_tr_np.shape[0]} trials)")
        elif args.model == "gvspd_learnable":
            model.gv.learnable_support.reinit_from_data(to_tensor(X_tr_np))
            model.gv._support_initialised = True
            print(f"  Support: learnable, init from global training data")

        print(f"  Train subjects: {[s for s in args.subjects if s != held_out]}")

        try:
            acc = run_training(model, X_tr, y_tr, X_te, y_te, args,
                               f"loso_{args.model}_held{held_out}")
            results[held_out] = acc
        except Exception as e:
            print(f"  x Failed: {e}")
            import traceback; traceback.print_exc()
            results[held_out] = None

    return results


# =============================================================
#  Main
# =============================================================
def main():
    global DTYPE, DEVICE

    p = argparse.ArgumentParser(description="GVSPD-Net BCI-IV-2a")
    p.add_argument("--model",
                    choices=["gvspd_fixed", "gvspd_learnable", "spdnet_baseline"],
                    default="gvspd_fixed")
    p.add_argument("--mode",
                    choices=["subject_specific", "subject_independent"],
                    default="subject_specific")
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--bimap-dims", nargs="+", type=int, default=None)
    p.add_argument("--pool", choices=["mean", "attention", "last"], default="mean")
    p.add_argument("--support",
                    choices=["log_cholesky", "matrix_exp", "eigenvalue", "bimap"],
                    default="log_cholesky")
    p.add_argument("--support-rank", type=int, default=None)
    p.add_argument("--lr-support", type=float, default=None)
    p.add_argument(
        "--n-windows", type=int, default=None,
        help="Number of temporal windows to average J over before SPDNet. "
             "None = use every time step (default). E.g. 50 averages T "
             "samples into 50 non-overlapping windows.",
    )
    p.add_argument("--tmin", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=6.0)
    p.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    # resolve device
    if args.device == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(args.device)

    _setup_cuda()
    DTYPE = torch.float64 if args.dtype == "float64" else torch.float32

    print("=" * 60)
    print("  GVSPD-Net -- BCI-IV-2a")
    print("=" * 60)
    print(f"  Model    : {args.model}")
    print(f"  Mode     : {args.mode}")
    print(f"  Subjects : {args.subjects}")
    print(f"  Epochs   : {args.epochs}  |  LR: {args.lr}")
    print(f"  t_range  : [{args.tmin}, {args.tmax}] s")
    print(f"  Device   : {DEVICE}")
    print(f"  Dtype    : {DTYPE}")
    if args.model == "gvspd_learnable":
        print(f"  Support  : {args.support}")
        if args.lr_support:
            print(f"  LR sup.  : {args.lr_support}")
    if args.n_windows is not None and args.model != "spdnet_baseline":
        print(f"  Windows  : {args.n_windows}")
    if DEVICE.type == "cuda":
        print(f"  GPU      : {torch.cuda.get_device_name(DEVICE)}")

    if args.mode == "subject_specific":
        results = run_subject_specific(args)
    else:
        results = run_subject_independent(args)

    # summary
    print(f"\n{'='*60}\n  SUMMARY  |  {args.model}  |  {args.mode}\n{'='*60}")
    accs = []
    for k in sorted(results):
        v = results[k]
        print(f"  Subject {k}: {f'{v:.4f}' if v is not None else 'FAILED'}")
        if v is not None:
            accs.append(v)
    if accs:
        print(f"\n  Mean: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/summary.txt", "w") as f:
        f.write(f"Model: {args.model}\nMode: {args.mode}\nDtype: {DTYPE}\n\n")
        for k in sorted(results):
            f.write(f"Subject {k}: {results[k]}\n")
        if accs:
            f.write(f"\nMean: {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n")

    print(f"\n  Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
