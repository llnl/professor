# Copyright 2026, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math

import pytest
import torch

from professor.utils import lr_schedule_factor


def test_constant_schedule_is_one_without_warmup():
    assert [lr_schedule_factor(step, 10) for step in range(12)] == [1.0] * 12


def test_linear_warmup():
    factors = [lr_schedule_factor(step, 100, "constant", warmup_steps=4) for step in range(6)]
    assert factors == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    assert lr_schedule_factor(1, 100, "cosine", warmup_steps=4) == 0.5


def test_cosine_schedule():
    total, warmup, lr_min = 110, 10, 0.01
    assert lr_schedule_factor(warmup, total, "cosine", warmup, lr_min) == pytest.approx(1.0)
    assert lr_schedule_factor(60, total, "cosine", warmup, lr_min) == pytest.approx(0.5 * (1.0 + lr_min))
    assert lr_schedule_factor(total, total, "cosine", warmup, lr_min) == pytest.approx(lr_min)
    assert lr_schedule_factor(2 * total, total, "cosine", warmup, lr_min) == pytest.approx(lr_min)
    factors = [lr_schedule_factor(step, total, "cosine", warmup, lr_min) for step in range(warmup, total + 1)]
    assert all(a > b for a, b in zip(factors, factors[1:]))


def test_cosine_schedule_matches_reference():
    """the schedule of the Rivus Rayleigh-Taylor study"""
    total, warmup, lr_min = 20000, 500, 0.01

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        f = (step - warmup) / max(1, total - warmup)
        return lr_min + (1 - lr_min) * 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    for step in [0, 1, 250, 499, 500, 501, 7000, 19999, 20000, 25000]:
        assert lr_schedule_factor(step, total, "cosine", warmup, lr_min) == pytest.approx(lr_at(step))


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
