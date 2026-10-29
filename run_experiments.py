"""Case study: image classification for medical imaging.

Trains and evaluates three models of increasing capacity on two MedMNIST v2 tasks:
  * PneumoniaMNIST - pediatric chest X-rays, normal vs. pneumonia (binary)
  * DermaMNIST     - dermatoscopic images (HAM10000), 7 skin-lesion classes

Models: PCA + logistic regression (classical baseline), a compact CNN trained from
scratch, and an ImageNet-pretrained ResNet-18 fine-tuned end to end (transfer learning).

Usage:  python run_experiments.py [--size 64] [--epochs-cnn 30] [--epochs-resnet 15] [--num-workers 4]
Outputs: outputs/results.json, outputs/figures/*.png, outputs/predictions/*.npz
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import plots
from src.data import load_dataset, make_loaders, class_weights
from src.gradcam import GradCAM
from src.metrics import compute_metrics, youden_threshold
from src.models import build_model
from src.train import train_model, predict

DATASETS = {
    "pneumoniamnist": {"title": "PneumoniaMNIST", "flip": False},
    "dermamnist": {"title": "DermaMNIST", "flip": True},
}
NEURAL = {"cnn": ("Simple CNN", 1e-3, "epochs_cnn"), "resnet18": ("ResNet-18 (TL)", 5e-4, "epochs_resnet")}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_baseline(splits):
    """PCA (95% variance, max 200 components) + class-balanced logistic regression on raw pixels."""
    flat = {s: x.reshape(len(x), -1).astype(np.float32) / 255.0 for s, (x, _) in splits.items()}
    clf = make_pipeline(StandardScaler(), PCA(n_components=200, random_state=0),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1))
    clf.fit(flat["train"], splits["train"][1])
    return {s: clf.predict_proba(flat[s]) for s in ("val", "test")}


def explain(model, splits, names, device, path, n=12, seed=0):
    """Grad-CAM on a stratified random sample of test images."""
    from src.data import MedImageDataset
    x, y = splits["test"]
    ds = MedImageDataset(x, y, train=False, augment_flip=False)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], int(np.ceil(n / len(classes))), replace=False)
                          for c in classes])[:n]
    cam = GradCAM(model, model.gradcam_layer)
    items = []
    for i in idx:
        heat, pred = cam(ds[i][0][None].to(device))
        short = lambda s: s.split(" ")[0][:12]
        items.append((x[i], heat, f"true: {short(names[y[i]])}\npred: {short(names[pred])}"))
    plots.gradcam_grid(items, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    ap.add_argument("--epochs-cnn", type=int, default=30)
    ap.add_argument("--epochs-resnet", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 4-8 on Linux)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS),
                    help="subset to (re)train; results of other datasets already in --out are kept")
    args = ap.parse_args()

    out, fig_dir, pred_dir = Path(args.out), Path(args.out) / "figures", Path(args.out) / "predictions"
    for d in (fig_dir, pred_dir, Path(args.data_root)):
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    for flag in args.datasets:
        cfg = DATASETS[flag]
        title = cfg["title"]
        print(f"\n=== {title} ===")
        splits, names = load_dataset(flag, args.size, args.data_root)
        n_cls = len(names)
        plots.sample_grid(splits["train"][0], splits["train"][1], names, fig_dir / f"samples_{flag}.png")
        res = {"classes": names,
               "n_images": {s: int(len(v[1])) for s, v in splits.items()},
               "class_counts": {s: np.bincount(v[1], minlength=n_cls).tolist() for s, v in splits.items()},
               "models": {}}

        model_probs = {}
        print("PCA + LogReg baseline ...")
        set_seed(args.seed)
        base = run_baseline(splits)
        model_probs["PCA + LogReg"] = base

        histories, trained = {}, {}
        weights = class_weights(splits["train"][1], n_cls)
        for key, (name, lr, ep_attr) in NEURAL.items():
            print(f"{name} ...")
            set_seed(args.seed)
            loaders = make_loaders(splits, args.batch_size, cfg["flip"], args.num_workers)
            model, hist = train_model(build_model(key, n_cls), loaders, device, getattr(args, ep_attr), lr, weights)
            histories[name], trained[key] = hist, model
            model_probs[name] = {s: predict(model, loaders[s], device)[0] for s in ("val", "test")}
            res.setdefault("training", {})[name] = hist
            torch.save(model.state_dict(), out / f"{flag}_{key}.pt")

        y_val, y_test = splits["val"][1], splits["test"][1]
        for name, p in model_probs.items():
            m = compute_metrics(y_test, p["test"])
            if n_cls == 2:
                thr = youden_threshold(y_val, p["val"][:, 1])
                m["youden_threshold"] = thr
                m["at_youden"] = {k: v for k, v in compute_metrics(y_test, p["test"], thr).items()
                                  if k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity",
                                           "ppv", "npv", "macro_f1", "confusion_matrix")}
            res["models"][name] = m
            np.savez(pred_dir / f"{flag}_{name.split(' ')[0].lower()}.npz", y=y_test, probs=p["test"])
            print(f"  {name:16s} AUC={m['auc']:.4f} [{m['auc_ci_low']:.3f}, {m['auc_ci_high']:.3f}]  "
                  f"acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  F1={m['macro_f1']:.4f}  ECE={m['ece']:.4f}")

        plots.training_curves(histories, f"{title}: training dynamics", fig_dir / f"training_{flag}.png")
        plots.roc_comparison(y_test, {n: (p["test"], res["models"][n]["auc"]) for n, p in model_probs.items()},
                             f"{title}: ROC on test set" + (" (micro-avg)" if n_cls > 2 else ""),
                             fig_dir / f"roc_{flag}.png")
        plots.reliability(y_test, {n: (p["test"], res["models"][n]["ece"]) for n, p in model_probs.items()
                                   if n != "PCA + LogReg"},
                          f"{title}: calibration", fig_dir / f"calibration_{flag}.png")
        for name in model_probs:
            plots.confusion(res["models"][name]["confusion_matrix"], names, f"{title} — {name}",
                            fig_dir / f"confusion_{flag}_{name.split(' ')[0].lower()}.png")
        if n_cls == 2:
            rn = res["models"]["ResNet-18 (TL)"]
            plots.threshold_tradeoff(y_test, model_probs["ResNet-18 (TL)"]["test"][:, 1],
                                     rn["youden_threshold"], fig_dir / f"threshold_{flag}.png")
        explain(trained["resnet18"], splits, names, device, fig_dir / f"gradcam_{flag}.png")
        with open(out / f"results_{flag}.json", "w") as f:  # saved per dataset so partial runs survive
            json.dump(res, f, indent=2)

    summarize(args, out, fig_dir)


def summarize(args, out, fig_dir):
    """Merges per-dataset results present in `out` into results.json and draws the cross-dataset figures."""
    results = {"config": vars(args), "datasets": {}}
    for flag in DATASETS:
        p = out / f"results_{flag}.json"
        if p.exists():
            results["datasets"][flag] = json.loads(p.read_text())
    loaded = []
    for flag in results["datasets"]:
        splits, names = load_dataset(flag, args.size, args.data_root)
        loaded.append((DATASETS[flag]["title"], names, splits))
    plots.class_distribution(loaded, fig_dir / "class_distribution.png")
    rows = [(DATASETS[f]["title"], name, m["auc"], m["auc_ci_low"], m["auc_ci_high"], m["balanced_accuracy"])
            for f, r in results["datasets"].items() for name, m in r["models"].items()]
    plots.model_comparison(rows, fig_dir / "model_comparison.png")
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results ({', '.join(results['datasets'])}) to {out / 'results.json'} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
