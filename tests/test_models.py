import torch
import pytest
from src.models.unet import UNet
from src.models.deeplabv3plus import DeepLabV3Plus


BATCH, C, H, W = 1, 3, 256, 256
NUM_CLASSES = 2


@pytest.fixture(scope="module")
def unet():
    # On désactive les poids préentraînés pour ne pas télécharger pendant les tests
    return UNet(encoder="resnet34", in_channels=C, num_classes=NUM_CLASSES, pretrained=False)


@pytest.fixture(scope="module")
def deeplab():
    model = DeepLabV3Plus(num_classes=NUM_CLASSES, in_channels=C, backbone="resnet50", pretrained=False)
    model.eval()
    return model


class TestUNet:

    def test_output_shape(self, unet):
        x = torch.randn(BATCH, C, H, W)
        out = unet(x)
        assert out.shape == (BATCH, NUM_CLASSES, H, W), f"Shape inattendue : {out.shape}"

    def test_no_nan_in_output(self, unet):
        x = torch.randn(BATCH, C, H, W)
        out = unet(x)
        assert not torch.isnan(out).any(), "NaN détecté dans la sortie UNet"

    def test_gradient_flows(self, unet):
        x = torch.randn(BATCH, C, H, W)
        out = unet(x)
        loss = out.sum()
        loss.backward()
        # Vérifie que le gradient a bien atteint les premiers poids du décodeur
        for name, param in unet.named_parameters():
            if "head" in name and param.grad is not None:
                assert not torch.isnan(param.grad).any()
                break

    def test_different_input_sizes(self, unet):
        # Le modèle doit accepter n'importe quelle taille divisible par 32
        for size in [128, 256, 512]:
            x = torch.randn(1, C, size, size)
            out = unet(x)
            assert out.shape == (1, NUM_CLASSES, size, size)


class TestDeepLabV3Plus:

    def test_output_shape(self, deeplab):
        x = torch.randn(BATCH, C, H, W)
        out = deeplab(x)
        assert out.shape == (BATCH, NUM_CLASSES, H, W), f"Shape inattendue : {out.shape}"

    def test_no_nan_in_output(self, deeplab):
        x = torch.randn(BATCH, C, H, W)
        out = deeplab(x)
        assert not torch.isnan(out).any(), "NaN détecté dans la sortie DeepLabV3+"

    def test_batch_size_1_train_mode(self, deeplab):
        deeplab.train()
        try:
            x = torch.randn(1, C, H, W)
            out = deeplab(x)
            assert out.shape[0] == 1
        finally:
            deeplab.eval()