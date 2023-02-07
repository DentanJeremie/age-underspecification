import torch.nn as nn
from torchvision.models import resnet50

pretrained_model = resnet50(weights="IMAGENET1K_V2")
class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 4),
        )

    def forward(self, x):
        logits = self.seq(x)
        return logits
