"""Tests des fonctions de perte :  """
import pytest
import torch
from src.losses import BCELoss, DiceLoss, TverskyLoss, ComboLoss, get_loss_function

def make_batch(perfect: bool = True, num_classes: int = 2, size: int = 8):
    # perfect=True -> prédit classe 1, False -> prédit classe 0 (faux)
    logits = torch.zeros(1, num_classes, size, size)
    if perfect:
        logits[:, 1, :, :] = 10.0
        logits[:, 0, :, :] = -10.0
    else:
        logits[:, 0, :, :] = 10.0
        logits[:, 1, :, :] = -10.0
    targets = torch.ones(1, size, size, dtype=torch.long)
    return logits, targets


class TestBCELoss:
    def test_perfect_prediction_is_low(self):
        loss_fn = BCELoss()
        logits, targets = make_batch(perfect=True)
        assert loss_fn(logits, targets).item() < 0.01

    def test_wrong_prediction_is_high(self):
        loss_fn = BCELoss()
        logits, targets = make_batch(perfect=False)
        assert loss_fn(logits, targets).item() > 5.0

    def test_uncertain_prediction_is_middle(self):
        loss_fn = BCELoss()
        logits = torch.zeros(1, 2, 8, 8)
        targets = torch.ones(1, 8, 8, dtype=torch.long)
        assert 0.5 < loss_fn(logits, targets).item() < 2.0

    def test_output_is_scalar(self):
        loss_fn = BCELoss()
        logits, targets = make_batch()
        assert loss_fn(logits, targets).shape == torch.Size([])

    def test_factory(self):
        assert isinstance(get_loss_function("bce"), BCELoss)


class TestDiceLoss:

    def test_perfect_prediction_near_zero(self):
        loss_fn = DiceLoss()
        logits, targets = make_batch(perfect=True)
        assert loss_fn(logits, targets).item() < 0.1

    def test_wrong_prediction_near_one(self):
        loss_fn = DiceLoss()
        logits, targets = make_batch(perfect=False)
        assert loss_fn(logits, targets).item() > 0.9

    def test_output_is_scalar(self):
        loss_fn = DiceLoss()
        logits, targets = make_batch()
        assert loss_fn(logits, targets).shape == torch.Size([])

    def test_loss_is_bounded(self):
        loss_fn = DiceLoss()
        for perfect in [True, False]:
            logits, targets = make_batch(perfect=perfect)
            val = loss_fn(logits, targets).item()
            assert 0.0 <= val <= 1.0


class TestTverskyLoss:


    def test_perfect_prediction_near_zero(self):
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        logits, targets = make_batch(perfect=True)
        assert loss_fn(logits, targets).item() < 0.05

    def test_wrong_prediction_near_one(self):

        loss_fn = TverskyLoss(alpha=0.5, beta=0.5)
        logits, targets = make_batch(perfect=False)
        assert loss_fn(logits, targets).item() > 0.9

    def test_asymmetric_weights(self):


        # beta élevé = pénalise plus les faux négatifs
        loss_fn = TverskyLoss(alpha=0.1, beta=0.9)
        logits, targets = make_batch(perfect=False)
        assert loss_fn(logits, targets).item() > 0.5

    def test_close_to_dice_when_symmetric(self):
        # Tversky(0.5, 0.5) devrait donner une valeur proche de Dice
        tversky = TverskyLoss(alpha=0.5, beta=0.5)
        dice = DiceLoss()
        logits, targets = make_batch(perfect=True)
        assert abs(tversky(logits, targets).item() - dice(logits, targets).item()) < 0.05


class TestComboLoss:

    def test_returns_scalar_and_dict(self):
        loss_fn = get_loss_function("bce_dice")
        logits, targets = make_batch()
        total, details = loss_fn(logits, targets)
        assert total.shape == torch.Size([])
        assert "bce" in details and "dice" in details

    def test_is_weighted_sum(self):
        logits, targets = make_batch()
        bce_val = BCELoss()(logits, targets).item()
        dice_val = DiceLoss()(logits, targets).item()
        total, _ = get_loss_function("bce_dice")(logits, targets)
        assert abs(total.item() - (0.5 * bce_val + 0.5 * dice_val)) < 1e-4

    def test_perfect_prediction_is_low(self):
        loss_fn = get_loss_function("bce_dice")
        logits, targets = make_batch(perfect=True)
        total, _ = loss_fn(logits, targets)
        assert total.item() < 0.05

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Perte inconnue"):
            get_loss_function("does_not_exist")