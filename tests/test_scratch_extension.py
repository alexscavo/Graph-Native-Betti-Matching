"""The continuation must preserve AdamW, LR progression and the next epoch shuffle."""
import copy
import unittest
import torch
from scripts.extend_scratch_geometry import restore_optimizer, capture_rng, restore_rng
from training.optim import build_optimizer, build_scheduler


class ExtensionTests(unittest.TestCase):
    def test_next_adamw_update_matches_uninterrupted_training(self):
        config = {'training': {'epochs': 300, 'warmup_epochs': 1,
            'optimizer': dict(name='adamw', base_lr=7e-5, weight_decay=1e-4,
                              epsilon=1e-8, betas=[.9,.999]),
            'scheduler': dict(name='polynomial', warmup_lr=1e-9, power=.9)}}
        torch.manual_seed(4)
        model = torch.nn.Linear(3, 1)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, 2)
        def update(net, opt, schedule):
            opt.zero_grad(); net(torch.ones(2,3)).square().mean().backward()
            opt.step(); schedule.step()
        for _ in range(3): update(model, optimizer, scheduler)
        saved = copy.deepcopy(dict(net=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), steps=3))
        restored = torch.nn.Linear(3, 1); restored.load_state_dict(saved['net'])
        new_optimizer = build_optimizer(config, restored)
        new_scheduler = build_scheduler(config, new_optimizer, 2)
        restore_optimizer(saved, new_optimizer, new_scheduler)
        update(model, optimizer, scheduler); update(restored, new_optimizer, new_scheduler)
        self.assertEqual(scheduler.get_last_lr(), new_scheduler.get_last_lr())
        for a,b in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        for a,b in zip(optimizer.state.values(), new_optimizer.state.values()):
            for key in a: torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

    def test_epoch_boundary_restores_next_shuffle_and_random_stream(self):
        loader = torch.utils.data.DataLoader(torch.arange(20), batch_size=4, shuffle=True,
                                            generator=torch.Generator().manual_seed(4))
        list(loader)  # Exhaust the epoch, as the continuation runner does.
        saved = capture_rng(loader)
        expected = torch.cat(list(loader)); expected_random = torch.rand(3)
        restore_rng(saved, loader)
        torch.testing.assert_close(torch.cat(list(loader)),expected,atol=0,rtol=0)
        torch.testing.assert_close(torch.rand(3),expected_random,atol=0,rtol=0)


if __name__ == '__main__': unittest.main()
