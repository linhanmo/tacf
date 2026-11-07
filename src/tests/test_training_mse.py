from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.training.stage1_pretrain import _stage1_loss_builder
from src.training.stage2_comm import _stage2_loss_builder
from src.training.stage3_aggregator import _stage3_loss_builder
from src.training.trainer import Trainer, TrainerConfig, build_mse_loss


@pytest.mark.parametrize("builder", [_stage2_loss_builder, _stage3_loss_builder])
def test_fusion_stages_use_only_final_prediction_mse(builder):
    cfg = TrainerConfig(lambda_nll=10.0, lambda_consensus=10.0, lambda_reject=10.0)
    prediction = torch.tensor([1.0, 3.0], requires_grad=True)
    y = torch.tensor([0.0, 1.0])
    # No sigma, rejection or auxiliary fields: these must never be accessed.
    out = SimpleNamespace(y_hat=prediction)
    loss = builder(cfg)(out, y)
    assert torch.equal(loss.total, F.mse_loss(prediction, y))
    loss.total.backward()
    assert torch.equal(prediction.grad, torch.tensor([1.0, 2.0]))
    assert loss.logging_dict() == {"mse": 2.5, "total": 2.5}


def test_stage1_averages_independent_specialist_mse():
    predictions = [
        torch.tensor([value], requires_grad=True) for value in (1.0, 2.0, 3.0)
    ]
    out = SimpleNamespace(
        specialists_out=SimpleNamespace(
            outputs={str(i): SimpleNamespace(mu=mu) for i, mu in enumerate(predictions)}
        )
    )
    cfg = TrainerConfig(lambda_nll=10.0, lambda_agent=10.0)
    loss = _stage1_loss_builder(None, cfg)(out, torch.zeros(1))
    assert torch.allclose(loss.total, torch.tensor(14.0 / 3.0))
    loss.total.backward()
    for mu in predictions:
        assert torch.allclose(mu.grad, 2.0 * mu.detach() / 3.0)


def test_mse_stays_fp32_with_half_precision_predictions():
    out = SimpleNamespace(y_hat=torch.tensor([300.0], dtype=torch.float16))
    loss = build_mse_loss(out, torch.zeros(1, dtype=torch.float16))
    assert loss.total.dtype == torch.float32
    assert loss.total.item() == 90000.0


def test_default_trainer_and_stage4_path_use_mse_only():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x, x_stamp, y):
            return SimpleNamespace(y_hat=x * self.weight)

    trainer = Trainer(Model(), TrainerConfig(device="cpu", amp=False))
    x = torch.ones(1, 2, 1)
    y = torch.zeros_like(x)
    _, loss, _, total = trainer._forward_loss((x, x, y, y))
    assert total.item() == 4.0
    assert total is loss.total
    total.backward()
    assert trainer.model.weight.grad.item() == 4.0
