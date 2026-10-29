"""Figures for the report. Colors follow a fixed categorical order so each model
keeps the same color across every chart."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLORS = {"PCA + LogReg": SERIES[0], "Simple CNN": SERIES[1], "ResNet-18 (TL)": SERIES[2]}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def class_distribution(datasets, path):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 3.8), squeeze=False)
    axes = axes[0]
    for ax, (title, names, splits) in zip(axes, datasets):
        split_names = ["train", "val", "test"]
        counts = np.array([np.bincount(splits[s][1], minlength=len(names)) for s in split_names])
        y = np.arange(len(names))
        h = 0.26
        for i, s in enumerate(split_names):
            ax.barh(y + (i - 1) * h, counts[i], height=h - 0.03, color=SERIES[i], label=s)
        short = [n if len(n) < 28 else n[:26] + "…" for n in names]
        ax.set_yticks(y, short)
        ax.invert_yaxis()
        ax.set_xlabel("Number of images")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        tot = counts[0]
        for yi, c in zip(y, tot):
            ax.text(c, yi - h, f" {c / tot.sum():.1%}", va="center", fontsize=8, color=INK2)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sample_grid(imgs, labels, names, path, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    k = len(names)
    fig, axes = plt.subplots(k, per_class, figsize=(per_class * 1.3, k * 1.4))
    axes = np.atleast_2d(axes)
    for c in range(k):
        idx = rng.choice(np.where(labels == c)[0], per_class, replace=False)
        for j, i in enumerate(idx):
            ax = axes[c, j]
            img = imgs[i]
            ax.imshow(img.squeeze(-1) if img.shape[-1] == 1 else img, cmap="gray")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        label = names[c] if len(names[c]) < 22 else names[c][:20] + "…"
        axes[c, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def training_curves(histories, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, h in histories.items():
        ep = np.arange(1, len(h["train_loss"]) + 1)
        c = MODEL_COLORS[name]
        axes[0].plot(ep, h["train_loss"], color=c, label=f"{name} — train")
        axes[0].plot(ep, h["val_loss"], color=c, linestyle="--", label=f"{name} — val")
        axes[1].plot(ep, h["val_auc"], color=c, label=name)
        axes[1].plot(h["best_epoch"], h["val_auc"][h["best_epoch"] - 1], "o", color=c, ms=8,
                     mec="white", mew=2)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation AUC", title="Validation AUC (dot = selected checkpoint)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def roc_comparison(y, model_probs, title, path):
    """Binary: ROC per model. Multi-class: micro-averaged one-vs-rest ROC per model."""
    fig, ax = plt.subplots(figsize=(5, 4.6))
    k = next(iter(model_probs.values()))[0].shape[1]
    for name, (p, auc) in model_probs.items():
        if k == 2:
            fpr, tpr, _ = roc_curve(y, p[:, 1])
        else:
            yb = label_binarize(y, classes=range(k))
            fpr, tpr, _ = roc_curve(yb.ravel(), p.ravel())
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1)
    ax.set(xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (sensitivity)",
           title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion(cm, names, title, path):
    cm = np.array(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True)
    k = len(names)
    fig, ax = plt.subplots(figsize=(1.0 * k + 2.5, 0.9 * k + 1.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    short = [n if len(n) < 16 else n[:14] + "…" for n in names]
    ax.set_xticks(range(k), short, rotation=35, ha="right")
    ax.set_yticks(range(k), short)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability(y, model_probs, title, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1, label="Perfect calibration")
    edges = np.linspace(0, 1, n_bins + 1)
    for name, (p, ece) in model_probs.items():
        conf, correct = p.max(1), (p.argmax(1) == y)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, "-o", color=MODEL_COLORS[name], ms=5, label=f"{name} (ECE {ece:.3f})")
    ax.set(xlabel="Predicted confidence", ylabel="Observed accuracy", title=title, xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def threshold_tradeoff(y, p_pos, thr_youden, path):
    thr = np.linspace(0.01, 0.99, 99)
    sens = [((p_pos >= t) & (y == 1)).sum() / (y == 1).sum() for t in thr]
    spec = [((p_pos < t) & (y == 0)).sum() / (y == 0).sum() for t in thr]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(thr, sens, color=SERIES[0], label="Sensitivity (pneumonia detected)")
    ax.plot(thr, spec, color=SERIES[1], label="Specificity (normal cleared)")
    ax.axvline(0.5, color=INK2, linestyle=":", linewidth=1)
    ax.axvline(thr_youden, color=INK, linestyle="--", linewidth=1)
    ax.text(0.5, 0.03, " default 0.5", color=INK2, fontsize=8)
    ax.text(thr_youden, 0.40, f" Youden (val) {thr_youden:.2f}", color=INK, fontsize=8)
    ax.set(xlabel="Decision threshold on P(pneumonia)", ylabel="Rate", ylim=(0, 1.02),
           title="ResNet-18 on PneumoniaMNIST test: threshold trade-off")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_comparison(rows, path):
    """rows: list of (dataset, model, auc, lo, hi, bal_acc)."""
    datasets = list(dict.fromkeys(r[0] for r in rows))
    models = list(MODEL_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(datasets))
    w = 0.26
    for metric_i, (ax, label) in enumerate(zip(axes, ["Test AUC (95% bootstrap CI)", "Test balanced accuracy"])):
        for i, m in enumerate(models):
            vals = [next(r for r in rows if r[0] == d and r[1] == m) for d in datasets]
            v = np.array([r[2] if metric_i == 0 else r[5] for r in vals])
            pos = x + (i - 1) * w
            ax.bar(pos, v, width=w - 0.03, color=MODEL_COLORS[m], label=m)
            if metric_i == 0:
                err = np.array([[r[2] - r[3], r[4] - r[2]] for r in vals]).T
                ax.errorbar(pos, v, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
            tops = [r[4] for r in vals] if metric_i == 0 else v
            for px, pv, top in zip(pos, v, tops):
                ax.text(px, top + 0.012, f"{pv:.3f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x, datasets)
        ax.set_ylim(0.4, 1.05)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gradcam_grid(items, path, ncols=6):
    """items: list of (image uint8 HWC, cam HxW, caption)."""
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 1.7, nrows * 3.6))
    axes = np.atleast_2d(axes)
    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        a_img, a_cam = axes[2 * r, c], axes[2 * r + 1, c]
        for a in (a_img, a_cam):
            a.set_xticks([]), a.set_yticks([])
            a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
        if k >= len(items):
            a_img.axis("off"), a_cam.axis("off")
            continue
        img, cam, cap = items[k]
        shown = img.squeeze(-1) if img.shape[-1] == 1 else img
        a_img.imshow(shown, cmap="gray")
        a_img.set_title(cap, fontsize=7, fontweight="normal")
        a_cam.imshow(shown, cmap="gray")
        a_cam.imshow(cam, cmap="jet", alpha=0.45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
