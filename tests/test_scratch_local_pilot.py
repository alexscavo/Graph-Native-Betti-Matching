"""Real training step: graph-only chords and active local residual supervision."""
import unittest
import torch
from models.local_spline_geometry import LocalSplineGeometry
from models.matcher import build_matcher
from scripts.controlled_spline_unfreeze import RoutedCriterion
from tests.test_frozen_spline_diagnostic import FrozenSplineTests
from training.engine import train_step
from scripts.train_spline_weight_pilot import assert_scratch_trainability


class ScratchLocalTests(unittest.TestCase):
    def test_real_decoder_legacy_freeze_is_allowed_but_active_freeze_is_not(self):
        from models.deformable_transformer import DeformableTransformerDecoderLayer
        layer = DeformableTransformerDecoderLayer(d_model=24,d_ffn=32,n_levels=1,
            n_heads=6,n_points=2,rln_attn=True,use_cuda_extension=False)
        model = torch.nn.Module()
        model.decoder = layer
        self.assertEqual(len(assert_scratch_trainability(model)),6)
        layer.linear1.weight.requires_grad_(False)
        with self.assertRaises(AssertionError):
            assert_scratch_trainability(model)

    def test_scratch_steps_update_bn_and_only_supervised_geometry(self):
        control_state = None
        for weight, route in ((0., 'tokens_only'), (1.4739989612791453, 'tokens_only'),
                              (1.4739989612791453, 'graph_only')):
            with self.subTest(weight=weight, route=route):
                cfg, model, _, batch = FrozenSplineTests().fixture()
                cfg['loss']['spline']['target'] = 'residual'
                cfg['loss']['weights']['spline'] = weight
                model.spline_head.local_geometry = LocalSplineGeometry(8, channels=3, samples=4)
                original_forward = model.forward
                def forward(image):
                    tokens, predictions, features = original_forward(image)
                    predictions.update(spline_local_features=features[:,:3,::2,::2,::2],
                                       spline_input_shape=tuple(image.shape[-3:]))
                    return tokens, predictions, features
                model.forward = forward
                criterion = RoutedCriterion(cfg,build_matcher(cfg),model.relation_embed,
                                            spline_head=model.spline_head,route=route)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
                before = model.encoder[1].running_mean.clone()
                for step in range(1,4):
                    losses = train_step(model,criterion,optimizer,scheduler,batch,cfg,'cpu',
                                        epoch=1,iteration=step,total_iterations=750000)
                    self.assertTrue(torch.isfinite(losses['total']))
                self.assertFalse(torch.equal(before,model.encoder[1].running_mean))
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                for layer in (model.spline_head.network[-1],model.spline_head.local_geometry.scorer[-1]):
                    changed = int(torch.count_nonzero(layer.weight)) + int(torch.count_nonzero(layer.bias))
                    if weight: self.assertGreater(changed,0)
                    else: self.assertEqual(changed,0)
                detector_state = {n:v.detach().clone() for n,v in model.state_dict().items()
                                  if not n.startswith('spline_head.')}
                if weight == 0.:
                    control_state = detector_state
                if route == 'graph_only':
                    for name, value in detector_state.items():
                        torch.testing.assert_close(value, control_state[name], atol=0, rtol=0)


if __name__=='__main__':unittest.main()
