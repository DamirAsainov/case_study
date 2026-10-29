"""Grad-CAM (Selvaraju et al., 2017) for explaining CNN predictions."""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = self.grads = None
        layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        """x: (1, C, H, W). Returns heatmap (H, W) in [0, 1] and the explained class."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        cls = int(logits.argmax(1)) if target is None else target
        self.model.zero_grad()
        logits[0, cls].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), cls
