import unittest
import torch
from torch import nn
from processing.vision.model import FretboardNet, fuse_encoder


class InferenceFusionTests(unittest.TestCase):
    def test_fused_encoder_preserves_eval_predictions(self):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        model = FretboardNet(pretrained=False).eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert module.running_mean is not None and module.running_var is not None
                module.running_mean.copy_(torch.randn_like(module.running_mean)*.1)
                module.running_var.copy_(torch.rand_like(module.running_var)+.5)
        image = torch.randn(1, 3, 64, 96)
        with torch.inference_mode():
            expected = model(image)
            fuse_encoder(model)
            actual = model(image)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in model.modules()))
        self.assertTrue(any(isinstance(m, nn.GroupNorm) for m in model.modules()))

    def test_fusion_rejects_training_mode(self):
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            fuse_encoder(FretboardNet(pretrained=False).train())
