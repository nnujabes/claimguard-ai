from __future__ import annotations
import torch
from torch import nn

class FusedAdapter(nn.Module):
    """Runnable fallback network. Replace only build_model() for the upstream FUSED architecture."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, image):
        features = self.encoder(image)
        return {"logits": self.cls_head(features).squeeze(1), "mask_logits": self.mask_head(features)}

def build_model(checkpoint: str | None = None):
    model = FusedAdapter()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=False)
    return model
