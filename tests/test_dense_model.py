import unittest
import torch
from processing.vision.model import FretboardNet, masked_loss


class DenseModelTests(unittest.TestCase):
    def test_forward_backward_and_partial_supervision(self):
        torch.set_num_threads(2)
        model=FretboardNet(pretrained=False).train()
        logits=model(torch.randn(1,3,64,96))
        self.assertEqual(tuple(logits.shape),(1,3,64,96))
        logits.retain_grad()
        target=torch.zeros_like(logits);target[:,0,15:40,10:70]=1
        valid=torch.zeros_like(logits);valid[:,0]=1
        loss=masked_loss(logits,target,valid);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.grad[:,1:].abs().sum().item(),0)
        self.assertGreater(logits.grad[:,0].abs().sum().item(),0)

    def test_empty_negative_has_finite_loss_and_gradient(self):
        logits=torch.zeros(1,3,16,16,requires_grad=True)
        loss=masked_loss(logits,torch.zeros_like(logits),torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logits.grad.sum().item(),0)
