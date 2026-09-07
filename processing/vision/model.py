"""FretboardNet: pretrained ResNet34 encoder, U-Net decoder, stride-2 dense heads."""
import pickle
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34, ResNet34_Weights
from processing.training.data import HEADS, image_tensor, letterbox
import cv2
import numpy as np


class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class FretboardNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layers = nn.ModuleList([encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4])
        self.decoders = nn.ModuleList([Block(512+256, 256), Block(256+128, 128), Block(128+64, 64), Block(64+64, 32)])
        self.head = nn.Conv2d(32, len(HEADS), 1)

    def train(self, mode=True):
        super().train(mode)
        # Stable pretrained running statistics with small high-resolution batches.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x):
        shape = x.shape[-2:]
        features = [self.stem(x)]
        x = self.pool(features[0])
        for layer in self.layers:
            x = layer(x); features.append(x)
        for block, skip in zip(self.decoders, reversed(features[:-1])):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], dim=1))
        return F.interpolate(self.head(x), size=shape, mode='bilinear', align_corners=False)


def masked_loss(logits, target, valid):
    # Unknown annotations must never become negative examples when datasets merge.
    positive = (target * valid).sum((0, 2, 3))
    negative = ((1-target) * valid).sum((0, 2, 3))
    weights = (negative / positive.clamp_min(1)).clamp(1, 30)[None, :, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=weights)
    counts = valid.sum((0, 2, 3))
    bce = (bce * valid).sum((0, 2, 3)) / counts.clamp_min(1)
    p = logits.sigmoid() * valid
    dice = 1 - (2*(p*target).sum((0, 2, 3))+1) / (p.sum((0, 2, 3))+positive+1)
    active = counts > 0
    return (bce[active] + dice[active]).mean() if active.any() else logits.sum() * 0


def load_checkpoint(path, device='cpu'):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('Not a dense state-dict checkpoint; use legacy_preview for YOLO OBB weights.') from error
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a dense checkpoint dictionary')
    if checkpoint.get('architecture') != 'fretboard-resnet34-unet-v1' or checkpoint.get('heads') != list(HEADS):
        raise ValueError('Expected a FretboardNet checkpoint. Use legacy_preview for YOLO OBB weights.')
    model = FretboardNet(pretrained=False)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


class DenseDetector:
    def __init__(self, path, device='cpu', size=None, optimize=True):
        self.device = device
        self.model, checkpoint = load_checkpoint(path, device)
        if optimize:
            fuse_encoder(self.model)
        self.size = size or checkpoint['config']['imgsz']
        if self.size < 64 or self.size % 32:
            raise ValueError('Image size must be >=64 and divisible by 32')
        if checkpoint.get('smoke_test'):
            warnings.warn('Smoke-test checkpoint: this model has not been trained for useful guitar detection.', stacklevel=2)

    @torch.inference_mode()
    def predict(self, image):
        boxed, (left, top, nw, nh) = letterbox(image, self.size)
        logits = self.model(image_tensor(boxed).unsqueeze(0).to(self.device))
        maps = logits.sigmoid()[0].cpu().numpy().transpose(1, 2, 0)
        maps = maps[top:top+nh, left:left+nw]
        return cv2.resize(maps, (image.shape[1], image.shape[0])).astype(np.float32)


def fuse_encoder(model):
    """Fold frozen encoder BatchNorm into convolutions for inference only."""
    if model.training:
        raise ValueError('Convolution fusion requires evaluation mode')
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    for module in model.modules():
        for conv_name, bn_name in (('conv1', 'bn1'), ('conv2', 'bn2')):
            conv, bn = getattr(module, conv_name, None), getattr(module, bn_name, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
        if isinstance(module, nn.Sequential):
            for index in range(len(module)-1):
                if isinstance(module[index], nn.Conv2d) and isinstance(module[index+1], nn.BatchNorm2d):
                    module[index] = fuse_conv_bn_eval(module[index], module[index+1])
                    module[index+1] = nn.Identity()
    return model

