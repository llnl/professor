# Copyright 2026, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import pytest
import torch

from professor.utils import lr_schedule_factor, lr_schedule_from_settings, lr_schedule_settings


def test_constant_schedule_is_one_without_warmup():
    assert [lr_schedule_factor(step, 10) for step in range(12)] == [1.0] * 12


def test_linear_warmup():
    factors = [lr_schedule_factor(step, 100, "constant", warmup_steps=4) for step in range(6)]
    assert factors == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    assert lr_schedule_factor(1, 100, "cosine", warmup_steps=4) == 0.5


def test_cosine_schedule():
    """the cosine starts at the full learning rate after the warm-up and reaches lr_min at the last step"""
    total, warmup, lr_min = 111, 10, 0.01
    assert lr_schedule_factor(warmup, total, "cosine", warmup, lr_min) == pytest.approx(1.0)
    assert lr_schedule_factor(60, total, "cosine", warmup, lr_min) == pytest.approx(0.5 * (1.0 + lr_min))
    assert lr_schedule_factor(total - 1, total, "cosine", warmup, lr_min) == pytest.approx(lr_min)
    assert lr_schedule_factor(total - 2, total, "cosine", warmup, lr_min) > lr_min
    assert lr_schedule_factor(2 * total, total, "cosine", warmup, lr_min) == pytest.approx(lr_min)
    factors = [lr_schedule_factor(step, total, "cosine", warmup, lr_min) for step in range(warmup, total)]
    assert all(a > b for a, b in zip(factors, factors[1:]))


def test_cosine_schedule_matches_reference():
    """the schedule of the Rivus Rayleigh-Taylor study, with lr_min at the last of its total steps"""
    total, warmup, lr_min = 20000, 500, 0.01

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        f = (step - warmup) / max(1, total - 1 - warmup)
        return lr_min + (1 - lr_min) * 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    for step in [0, 1, 250, 499, 500, 501, 7000, 19998, 19999, 20000, 25000]:
        assert lr_schedule_factor(step, total, "cosine", warmup, lr_min) == pytest.approx(lr_at(step))


def test_cosine_decay_from_a_restart_step():
    """decay_start and decay_from start the cosine from the learning rate at a restart step"""
    total, warmup, lr_min = 100, 10, 0.1
    f0 = lr_schedule_factor(40, total, "cosine", warmup, lr_min)
    factors = [lr_schedule_factor(step, 200, "cosine", warmup, lr_min, 40, f0) for step in range(40, 200)]
    assert factors[0] == pytest.approx(f0)
    assert factors[-1] == pytest.approx(lr_min)
    assert all(a > b for a, b in zip(factors, factors[1:]))
    # the warm-up does not depend on the decay
    assert lr_schedule_factor(3, 200, "cosine", warmup, lr_min, 40, f0) == pytest.approx(0.4)


def test_settings_defaults_and_given():
    settings = lr_schedule_settings({"lr_schedule": None, "warmup_steps": None, "lr_min": None}, 50)
    assert settings == {
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "lr_min": 0.01,
        "total_steps": 50,
        "decay_start": None,
        "decay_from": 1.0,
    }
    settings = lr_schedule_settings({"lr_schedule": "cosine", "warmup_steps": 5, "lr_min": None}, 50)
    assert (settings["lr_schedule"], settings["warmup_steps"], settings["lr_min"]) == ("cosine", 5, 0.01)


def test_restart_uses_the_saved_settings():
    """a restart without the schedule arguments continues the saved schedule"""
    saved = lr_schedule_settings({"lr_schedule": "cosine", "warmup_steps": 5, "lr_min": 0.1}, 100)
    restarted = lr_schedule_settings({}, 100, saved, restart_step=40)
    assert restarted == saved
    restarted = lr_schedule_settings({"lr_schedule": "cosine", "lr_min": 0.1}, 100, saved, restart_step=40)
    assert restarted == saved


def test_restart_rejects_different_settings():
    saved = lr_schedule_settings({"lr_schedule": "cosine", "warmup_steps": 5, "lr_min": 0.1}, 100)
    with pytest.raises(ValueError, match="--lr_min 0.2 differs from 0.1"):
        lr_schedule_settings({"lr_min": 0.2}, 100, saved, restart_step=40)
    with pytest.raises(ValueError, match="--lr_schedule constant differs from cosine"):
        lr_schedule_settings({"lr_schedule": "constant"}, 100, saved, restart_step=40)


@pytest.mark.parametrize("new_total", [60, 100, 200])
def test_restart_with_a_different_number_of_steps_does_not_jump(new_total):
    """the learning rate at the restart step stays, and decays to lr_min at the new last step"""
    saved = lr_schedule_settings({"lr_schedule": "cosine", "warmup_steps": 5, "lr_min": 0.1}, 100)
    restarted = lr_schedule_settings({}, new_total, saved, restart_step=40)
    before = lr_schedule_from_settings(40, saved)
    assert lr_schedule_from_settings(40, restarted) == pytest.approx(before)
    assert lr_schedule_from_settings(new_total - 1, restarted) == pytest.approx(0.1)
    factors = [lr_schedule_from_settings(step, restarted) for step in range(40, new_total)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))
    # and a second restart with the same number of steps keeps the decay of the first
    again = lr_schedule_settings({}, new_total, restarted, restart_step=50)
    assert again == restarted


def test_restart_in_the_warmup_with_a_different_number_of_steps():
    """the warm-up does not depend on the number of steps, so a restart in it keeps the plain schedule"""
    saved = lr_schedule_settings({"lr_schedule": "cosine", "warmup_steps": 50, "lr_min": 0.1}, 100)
    restarted = lr_schedule_settings({}, 200, saved, restart_step=20)
    assert restarted == lr_schedule_settings({"lr_schedule": "cosine", "warmup_steps": 50, "lr_min": 0.1}, 200)


def test_unknown_schedule():
    with pytest.raises(ValueError, match="Unrecognized learning rate schedule"):
        lr_schedule_factor(5, 10, "step")


def _optimizer_and_scheduler(total=20):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def lr_lambda(step):
        return lr_schedule_factor(step, total, "cosine", 4, 0.1)

    return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_step(model, optimizer, scheduler):
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def test_scheduler_follows_schedule_and_resumes_from_its_state():
    """the learning rate of each optimizer step follows the schedule, also across a restart"""
    model, optimizer, scheduler = _optimizer_and_scheduler()
    lrs = []
    for _ in range(7):
        lrs.append(optimizer.param_groups[0]["lr"])
        _train_step(model, optimizer, scheduler)
    assert lrs == pytest.approx([1e-3 * lr_schedule_factor(step, 20, "cosine", 4, 0.1) for step in range(7)])

    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    model2, optimizer2, scheduler2 = _optimizer_and_scheduler()
    optimizer2.load_state_dict(state["optimizer"])
    scheduler2.load_state_dict(state["scheduler"])
    for _ in range(5):
        assert optimizer2.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
        _train_step(model, optimizer, scheduler)
        _train_step(model2, optimizer2, scheduler2)
    assert scheduler2.last_epoch == 12
