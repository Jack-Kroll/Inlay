"""FretboardNet: pretrained ResNet34 encoder, U-Net decoder, stride-2 dense heads.

The network outputs three per-pixel heatmaps (see ``HEADS``): neck, fret wire
and nut. They are independent sigmoid channels, not a softmax, because a fret
wire pixel is also a neck pixel.
"""
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
    """Two 3x3 conv -> GroupNorm -> SiLU layers: the basic decoder unit."""
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class FretboardNet(nn.Module):
    """U-Net: ResNet34 shrinks the image into features, the decoder grows them back.

    Skip connections feed each decoder stage the encoder features at the same
    resolution, which keeps thin fret wires sharp.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        # Encoder: ResNet34 minus its classification layers.
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # 1/2 resolution, 64 ch
        self.pool = encoder.maxpool
        self.layers = nn.ModuleList([encoder.layer1,   # 1/4,  64 ch
                                     encoder.layer2,   # 1/8,  128 ch
                                     encoder.layer3,   # 1/16, 256 ch
                                     encoder.layer4])  # 1/32, 512 ch
        # Decoder: each block takes (upsampled features + skip features).
        self.decoders = nn.ModuleList([Block(512+256, 256), Block(256+128, 128),
                                       Block(128+64, 64), Block(64+64, 32)])
        # 1x1 conv to one logit per head (neck, fret, nut).
        self.head = nn.Conv2d(32, len(HEADS), 1)

    def train(self, mode=True):
        super().train(mode)
        # Keep the pretrained BatchNorm statistics frozen. Training uses tiny
        # high-resolution batches, which would give noisy running statistics.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x):
        shape = x.shape[-2:]
        # Encode, remembering the output of each stage for skip connections.
        features = [self.stem(x)]
        x = self.pool(features[0])
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        # Decode: upsample, concatenate the matching skip, refine. The deepest
        # feature map is the starting point, so it is not reused as a skip.
        for block, skip in zip(self.decoders, reversed(features[:-1])):
            upsampled = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([upsampled, skip], dim=1))
        # Logits at stride 2, resized up to the input size.
        return F.interpolate(self.head(x), size=shape, mode='bilinear', align_corners=False)


def masked_loss(logits, target, valid):
    """Weighted binary cross-entropy + Dice, counted only where labels exist.

    ``valid`` marks which pixels/heads were actually annotated. Datasets are
    merged from sources that label different heads, so an unlabelled pixel
    must never be treated as a negative example.
    """
    # Per-head counts of labelled positive and negative pixels.
    positive = (target * valid).sum((0, 2, 3))
    negative = ((1-target) * valid).sum((0, 2, 3))
    # Wires are a tiny fraction of pixels; up-weight positives (capped at 30x)
    # so the model can't win by predicting "no wire" everywhere.
    weights = (negative / positive.clamp_min(1)).clamp(1, 30)[None, :, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none', pos_weight=weights)
    counts = valid.sum((0, 2, 3))
    bce = (bce * valid).sum((0, 2, 3)) / counts.clamp_min(1)
    # Dice loss measures overlap, which also helps thin, sparse targets.
    p = logits.sigmoid() * valid
    dice = 1 - (2*(p*target).sum((0, 2, 3))+1) / (p.sum((0, 2, 3))+positive+1)
    # Average only over heads with any labels in this batch.
    active = counts > 0
    return (bce[active] + dice[active]).mean() if active.any() else logits.sum() * 0


def load_checkpoint(path, device='cpu'):
    """Load a trained FretboardNet, refusing files that aren't one."""
    try:
        # weights_only=True avoids executing arbitrary pickled code.
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('Not a FretboardNet state-dict checkpoint.') from error
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a dense checkpoint dictionary')
    if checkpoint.get('architecture') != 'fretboard-resnet34-unet-v1' or checkpoint.get('heads') != list(HEADS):
        raise ValueError('Expected a FretboardNet checkpoint.')
    model = FretboardNet(pretrained=False)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


class DenseDetector:
    """Inference wrapper: BGR frame in, HxWx3 heatmaps (0..1) out at frame size."""

    def __init__(self, path, device='cpu', size=None, optimize=True):
        self.device = device
        self.model, checkpoint = load_checkpoint(path, device)
        if optimize:
            fuse_encoder(self.model)
        # Default to the resolution the model was trained at.
        self.size = size or checkpoint['config']['imgsz']
        if self.size < 64 or self.size % 32:
            raise ValueError('Image size must be >=64 and divisible by 32')
        if checkpoint.get('smoke_test'):
            warnings.warn('Smoke-test checkpoint: this model has not been trained for useful guitar detection.', stacklevel=2)

    @torch.inference_mode()
    def predict(self, image):
        # Pad to a square (keeping aspect ratio), run the network, then crop
        # the padding off and resize back to the original frame.
        boxed, (left, top, nw, nh) = letterbox(image, self.size)
        logits = self.model(image_tensor(boxed).unsqueeze(0).to(self.device))
        maps = logits.sigmoid()[0].cpu().numpy().transpose(1, 2, 0)
        maps = maps[top:top+nh, left:left+nw]
        return cv2.resize(maps, (image.shape[1], image.shape[0])).astype(np.float32)


def fuse_encoder(model):
    """Fold frozen encoder BatchNorm into the preceding convolutions.

    A BatchNorm in eval mode is just a fixed scale and shift, so it can be baked
    into the conv weights. Same output, fewer ops: a free inference speed-up.
    """
    if model.training:
        raise ValueError('Convolution fusion requires evaluation mode')
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    for module in model.modules():
        # ResNet blocks name their pairs conv1/bn1 and conv2/bn2.
        for conv_name, bn_name in (('conv1', 'bn1'), ('conv2', 'bn2')):
            conv, bn = getattr(module, conv_name, None), getattr(module, bn_name, None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(module, conv_name, fuse_conv_bn_eval(conv, bn))
                setattr(module, bn_name, nn.Identity())
        # Sequential containers (the stem, downsample paths) hold them side by side.
        if isinstance(module, nn.Sequential):
            for index in range(len(module)-1):
                conv, bn = module[index], module[index+1]
                if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                    module[index] = fuse_conv_bn_eval(conv, bn)
                    module[index+1] = nn.Identity()
    return model
