import pytest
import torch

from src.model import (
    ResNet18SpectrogramClassifier,
    SmallSpectrogramCNN,
    build_model,
)


def test_small_cnn_forward_pass() -> None:
    model = SmallSpectrogramCNN(num_classes=4)

    output = model(torch.randn(4, 1, 224, 224))

    assert output.shape == (4, 4)


@pytest.mark.parametrize("channels", [1, 3])
def test_resnet18_forward_pass_accepts_supported_channels(channels: int) -> None:
    model = ResNet18SpectrogramClassifier(num_classes=4, pretrained=False)

    output = model(torch.randn(2, channels, 224, 224))

    assert output.shape == (2, 4)


def test_build_model_returns_supported_models() -> None:
    small_model = build_model("small_cnn")
    resnet_model = build_model("resnet18", pretrained=False)

    assert isinstance(small_model, SmallSpectrogramCNN)
    assert isinstance(resnet_model, ResNet18SpectrogramClassifier)


def test_build_model_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="unsupported model_name"):
        build_model("unknown")


def test_freeze_backbone_only_leaves_classifier_trainable() -> None:
    model = ResNet18SpectrogramClassifier(
        pretrained=False,
        freeze_backbone=True,
    )

    backbone_parameters = [
        parameter
        for name, parameter in model.backbone.named_parameters()
        if not name.startswith("fc.")
    ]
    classifier_parameters = list(model.backbone.fc.parameters())

    assert backbone_parameters
    assert classifier_parameters
    assert all(not parameter.requires_grad for parameter in backbone_parameters)
    assert all(parameter.requires_grad for parameter in classifier_parameters)


@pytest.mark.parametrize(
    "model_class", [SmallSpectrogramCNN, ResNet18SpectrogramClassifier]
)
def test_models_reject_invalid_num_classes(
    model_class: type[torch.nn.Module],
) -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        model_class(num_classes=1)  # type: ignore[call-arg]
