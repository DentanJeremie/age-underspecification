import torch.nn as nn
from torchvision.models import resnet50, vgg19
import torchvision


pretrained_resnet50 = resnet50(weights="IMAGENET1K_V2")
class AgeModelRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            pretrained_resnet50,
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        logits = self.seq(x)
        return logits

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_head(self):
        for name, param in self.named_parameters():
            if 'seq.0.fc' in name or 'seq.1' in name:
                param.requires_grad = True

    def unfreeze_last_conv_plus(self):
        self.unfreeze_head()
        for name, param in self.named_parameters():
            if 'seq.0.layer4' in name:
                param.requires_grad = True

    def unfreeze_two_last_conv_plus(self):
        self.unfreeze_last_conv_plus()
        for name, param in self.named_parameters():
            if 'seq.0.layer3' in name:
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def check_requires_grad(self):
        for name, para in self.named_parameters():
            print("-"*20)
            print(f"name: {name}")
            print(f"requires_grad: {para.requires_grad}")

pretrained_vgg19 = vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
class AgeModelVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            pretrained_vgg19,
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        logits = self.seq(x)
        return logits

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_head(self):
        for name, param in self.named_parameters():
            if 'classifier' in name or 'seq.1' in name:
                param.requires_grad = True

    def unfreeze_last_conv_plus(self):
        self.unfreeze_head()
        for name, param in self.named_parameters():
            if '28' in name or '30' in name or '32' in name or '34' in name:
                param.requires_grad = True

    def unfreeze_two_last_conv_plus(self):
        self.unfreeze_last_conv_plus()
        for name, param in self.named_parameters():
            if '19' in name or '21' in name or '23' in name or '25' in name:
                param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def check_requires_grad(self):
        for name, para in self.named_parameters():
            print("-"*20)
            print(f"name: {name}")
            print(f"requires_grad: {para.requires_grad}")