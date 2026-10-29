"""Training loop with class-weighted loss, cosine LR schedule, AMP and best-on-validation checkpointing."""
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from .metrics import auc_score


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train_model(model, loaders, device, epochs, lr, weights, log=print):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=len(loaders["train"]), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    best_auc, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        p, yv = predict(model, loaders["val"], device)
        val_loss = nn.functional.nll_loss(torch.log(torch.tensor(p) + 1e-8), torch.tensor(yv)).item()
        val_auc = auc_score(yv, p)
        val_acc = float((p.argmax(1) == yv).mean())
        for k, v in zip(history, (total / n, val_loss, val_auc, val_acc)):
            history[k].append(v)
        if val_auc > best_auc:
            best_auc, best_state, best_ep = val_auc, copy.deepcopy(model.state_dict()), ep
        log(f"  epoch {ep:2d}/{epochs}  train_loss={total / n:.4f}  val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.4f}  val_acc={val_acc:.4f}")
    model.load_state_dict(best_state)
    history["best_epoch"] = best_ep
    history["train_time_s"] = time.time() - t0
    return model, history
