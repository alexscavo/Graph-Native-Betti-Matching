"""Focused checks for the reactivated reference domain objective."""

import copy
from pathlib import Path
import unittest

import torch
from torch import nn

from configs import ConfigError, load_config, validate_config
from data.loaders.mixed import compose_source_target
from models.checkpoint import checkpoint_compatibility
from models.domain_adaptation import DomainDiscriminator
from models.matcher import build_matcher
from models.relationformer import RelationFormer
from training.losses import GraphCriterion
from training.optim import build_optimizer


def _config():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "pretrain_mixed.yaml",
        environment={
            "GNBM_OUTPUT_DIR": "/outputs",
            "PLANTS_DATASET": "/plants",
            "SYNTHETIC_MRI_DATASET": "/ixi",
        },
    )
    config = copy.deepcopy(config)
    config["model"]["decoder"].update(hidden_dim=12, object_queries=4, relation_tokens=1)
    config["domain_adaptation"]["enabled"] = True
    config["loss"]["enabled"].append("domain")
    return config


class DomainAdaptationTests(unittest.TestCase):
    def test_model_emits_both_reference_domain_predictions(self):
        config = _config()
        config["model"]["decoder"]["hidden_dim"] = 48
        config["domain_adaptation"]["hidden_size"] = 16

        class Encoder(nn.Module):
            num_features = 48

            def __init__(self):
                super().__init__()
                self.proj = nn.Conv3d(1, 48, 1)

            def forward(self, volumes):
                return self.proj(volumes)

        class Decoder(nn.Module):
            def forward(self, projected, mask, queries, positions):
                return projected.mean(dim=(2, 3, 4))[:, None, :] + queries[None]

        model = RelationFormer(Encoder(), Decoder(), config).eval()
        with torch.no_grad():
            tokens, predictions, features = model(torch.rand(2, 1, 2, 2, 2))
        self.assertEqual(tokens.shape, (2, 5, 48))
        self.assertEqual(predictions["domain_image_log_probs"].shape, (16, 2))
        self.assertEqual(predictions["domain_instance_log_probs"].shape, (2, 2))
        self.assertEqual(features.shape, (2, 48, 2, 2, 2))
        optimizer = build_optimizer(config, model)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[1]["lr"], 7e-5)

    def test_reference_domain_loss_is_in_total_and_reaches_features(self):
        config = _config()
        config["loss"]["enabled"] = ["domain"]
        relation = nn.Linear(36, 2)
        criterion = GraphCriterion(config, build_matcher(config), relation)
        image_discriminator = DomainDiscriminator(12, hidden_size=16)
        instance_discriminator = DomainDiscriminator(60, hidden_size=16)
        features = torch.randn(2, 12, requires_grad=True)
        tokens = features[:, None, :].expand(2, 5, 12).contiguous()
        image_reversed, image_plain = image_discriminator(
            features.repeat_interleave(3, dim=0), 0.7
        )
        instance_reversed, instance_plain = instance_discriminator(
            tokens.flatten(start_dim=1), 0.7
        )
        predictions = {
            "pred_logits": torch.randn(2, 4, 2, requires_grad=True),
            "pred_nodes": torch.rand(2, 4, 6, requires_grad=True),
            "domain_image_log_probs": image_reversed,
            "domain_image_plain_log_probs": image_plain,
            "domain_instance_log_probs": instance_reversed,
            "domain_instance_plain_log_probs": instance_plain,
        }
        targets = {
            "nodes": [torch.tensor([[0.2, 0.3, 0.4]]), torch.tensor([[0.6, 0.7, 0.8]])],
            "edges": [torch.empty((0, 2), dtype=torch.long) for _ in range(2)],
            "domains": torch.tensor([0, 1]),
        }
        losses = criterion(tokens, predictions, targets)
        self.assertTrue(torch.allclose(losses["total"], losses["domain"]))
        self.assertTrue(torch.allclose(losses["domain_weighted"], losses["domain"]))
        losses["total"].backward()
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertGreater(float(image_discriminator.net[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(instance_discriminator.net[0].weight.grad.abs().sum()), 0.0)

    def test_domain_config_requires_mixture_and_optimized_loss(self):
        config = _config()
        validate_config(config)
        config["domain_adaptation"]["class_weights"] = ["bad", 0.95]
        with self.assertRaisesRegex(ConfigError, "class_weights"):
            validate_config(config)
        config["domain_adaptation"]["class_weights"] = [0.05, 0.95]
        config["loss"]["enabled"].remove("domain")
        with self.assertRaisesRegex(ConfigError, "requires domain in loss.enabled"):
            validate_config(config)
        config["loss"]["enabled"].append("domain")
        del config["data"]["datasets"]["plants"]
        with self.assertRaisesRegex(ConfigError, "requires a source and target"):
            validate_config(config)

    def test_reference_sampler_uses_combined_epoch_length(self):
        source = torch.utils.data.TensorDataset(torch.arange(10))
        target = torch.utils.data.TensorDataset(torch.arange(2))
        dataset, sampler = compose_source_target(
            source, target, balance_source_target=True, epoch_size="combined"
        )
        self.assertEqual(len(dataset), 12)
        self.assertEqual(len(sampler), 12)
        self.assertEqual(sampler.weights[-1], 5.0)

    def test_domain_checkpoint_keys_are_active_only_with_heads(self):
        with_domain = nn.Module()
        with_domain.backbone_domain_discriminator = nn.Linear(2, 2)
        checkpoint = {"model": with_domain.state_dict()}
        self.assertTrue(checkpoint_compatibility(with_domain, checkpoint).compatible)
        without_domain = nn.Module()
        report = checkpoint_compatibility(without_domain, checkpoint)
        self.assertEqual(len(report.ignored_removed), 2)
        self.assertTrue(report.compatible)


if __name__ == "__main__":
    unittest.main()
