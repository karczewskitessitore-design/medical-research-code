from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import UNet


class ConvBlock(nn.Module):
    """Two-convolution block used by the clean-room journal baselines."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class Decoder2D(nn.Module):
    def __init__(self, channels: tuple[int, ...], out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for index in range(len(channels) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(channels[index], channels[index - 1], 2, 2))
            self.blocks.append(ConvBlock(channels[index - 1] * 2, channels[index - 1], dropout))
        self.output = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        value = features[-1]
        for up, block, skip in zip(self.up, self.blocks, reversed(features[:-1])):
            value = up(value)
            if value.shape[-2:] != skip.shape[-2:]:
                value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = block(torch.cat((skip, value), dim=1))
        return self.output(value)


class ClassicUNet2D(nn.Module):
    """Classic 2D U-Net used for Daniel and Damiano clean-room replications."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoders = []
        previous = in_channels
        for channel in channels:
            encoders.append(ConvBlock(previous, channel, dropout))
            previous = channel
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(2)
        self.decoder = Decoder2D(channels, out_channels, dropout)

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        value = image
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            features.append(value)
            if index < len(self.encoders) - 1:
                value = self.pool(value)
        return features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(image))


class DualHeadUNet2D(nn.Module):
    """Shared encoder with two independent decoders for CMMT-style training."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoders = []
        previous = in_channels
        for channel in channels:
            encoders.append(ConvBlock(previous, channel, dropout))
            previous = channel
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(2)
        self.decoder_a = Decoder2D(channels, out_channels, dropout)
        self.decoder_b = Decoder2D(channels, out_channels, dropout)

    def encode(self, image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        value = image
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            features.append(value)
            if index < len(self.encoders) - 1:
                value = self.pool(value)
        return features

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(image)
        return self.decoder_a(features), self.decoder_b(features)


class UNetPlusPlus2D(nn.Module):
    """Nested skip-path UNet++ with optional deep supervision."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...],
        deep_supervision: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        previous = in_channels
        for channel in channels:
            self.encoders.append(ConvBlock(previous, channel, dropout))
            previous = channel
        self.nested = nn.ModuleDict()
        for depth in range(1, len(channels)):
            for level in range(len(channels) - depth):
                incoming = channels[level] * depth + channels[level + 1]
                self.nested[f"{level}_{depth}"] = ConvBlock(incoming, channels[level], dropout)
        self.outputs = nn.ModuleList(
            nn.Conv2d(channels[0], out_channels, 1) for _ in range(1, len(channels))
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        nodes: dict[tuple[int, int], torch.Tensor] = {}
        value = image
        for level, encoder in enumerate(self.encoders):
            value = encoder(value)
            nodes[(level, 0)] = value
            if level < len(self.encoders) - 1:
                value = self.pool(value)
        predictions = []
        for depth in range(1, len(self.channels)):
            for level in range(len(self.channels) - depth):
                upsampled = F.interpolate(
                    nodes[(level + 1, depth - 1)],
                    size=nodes[(level, 0)].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                skip_nodes = [nodes[(level, prior)] for prior in range(depth)]
                nodes[(level, depth)] = self.nested[f"{level}_{depth}"](
                    torch.cat((*skip_nodes, upsampled), dim=1)
                )
            predictions.append(self.outputs[depth - 1](nodes[(0, depth)]))
        if self.deep_supervision:
            return tuple(predictions)
        return predictions[-1]


def _monai_unet(config: dict[str, Any]) -> nn.Module:
    channels = tuple(int(value) for value in config["channels"])
    return UNet(
        spatial_dims=2,
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        channels=channels,
        strides=(2,) * (len(channels) - 1),
        num_res_units=int(config.get("residual_units", 2)),
        norm="INSTANCE",
    )


def _classic_unet(config: dict[str, Any]) -> nn.Module:
    options = config.get("options", {})
    return ClassicUNet2D(
        int(config["in_channels"]),
        int(config["out_channels"]),
        tuple(int(value) for value in config["channels"]),
        float(options.get("dropout", 0.0)),
    )


def _unetpp(config: dict[str, Any]) -> nn.Module:
    options = config.get("options", {})
    return UNetPlusPlus2D(
        int(config["in_channels"]),
        int(config["out_channels"]),
        tuple(int(value) for value in config["channels"]),
        bool(options.get("deep_supervision", True)),
        float(options.get("dropout", 0.0)),
    )


def _cmmt(config: dict[str, Any]) -> nn.Module:
    options = config.get("options", {})
    return DualHeadUNet2D(
        int(config["in_channels"]),
        int(config["out_channels"]),
        tuple(int(value) for value in config["channels"]),
        float(options.get("dropout", 0.0)),
    )


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "monai_unet": _monai_unet,
    "daniel_unet": _classic_unet,
    "damiano_unet": _classic_unet,
    "unetpp": _unetpp,
    "tcsm_unet": _monai_unet,
    "cmmt_net": _cmmt,
}


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))


def build_model(config: dict[str, Any]) -> nn.Module:
    name = str(config.get("name", "monai_unet"))
    if name in {"nnunetv2_2d", "nnunetv2_3d_fullres"}:
        raise ValueError(f"{name} is executed through the external nnU-Net adapter")
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; available: {registered_models()}") from error
    return factory(config)


def logits_heads(output: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return output if isinstance(output, tuple) else (output,)


def primary_logits(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    heads = logits_heads(output)
    return heads[0] if len(heads) == 1 else torch.stack(heads, dim=0).mean(dim=0)


def create_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    student_state = student.state_dict()
    for name, teacher_value in teacher.state_dict().items():
        student_value = student_state[name]
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)
