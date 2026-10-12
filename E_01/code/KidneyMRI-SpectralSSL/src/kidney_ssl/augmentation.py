from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kidney_ssl.bilateral import swap_bilateral_labels


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    lr_flip_probability: float = 0.5
    rotation_degrees: float = 15.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    translation_fraction: float = 0.10
    gamma_min: float = 0.7
    gamma_max: float = 1.5
    contrast_min: float = 0.8
    contrast_max: float = 1.2
    bias_strength: float = 0.3
    noise_std: float = 0.03

    @classmethod
    def from_mapping(cls, values: dict | None) -> "AugmentationConfig":
        values = values or {}
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})


def _uniform(
    batch: int,
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.empty(batch, device=device).uniform_(low, high, generator=generator)


def _spatial_transform(
    image: torch.Tensor,
    labels: torch.Tensor | None,
    config: AugmentationConfig,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch = image.shape[0]
    device = image.device
    radians = _uniform(
        batch, -config.rotation_degrees, config.rotation_degrees, device, generator
    ) * torch.pi / 180.0
    scale = _uniform(batch, config.scale_min, config.scale_max, device, generator)
    tx = _uniform(
        batch, -2.0 * config.translation_fraction, 2.0 * config.translation_fraction, device, generator
    )
    ty = _uniform(
        batch, -2.0 * config.translation_fraction, 2.0 * config.translation_fraction, device, generator
    )
    cosine = torch.cos(radians) / scale
    sine = torch.sin(radians) / scale
    theta = image.new_zeros((batch, 2, 3))
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    image = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    if labels is not None:
        label_shape = labels.shape
        label_tensor = labels[:, None].float() if labels.ndim == 3 else labels.float()
        labels = F.grid_sample(
            label_tensor, grid, mode="nearest", padding_mode="zeros", align_corners=False
        ).long()
        if len(label_shape) == 3:
            labels = labels[:, 0]
    return image, labels


def _intensity_transform(
    image: torch.Tensor,
    config: AugmentationConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    batch = image.shape[0]
    device = image.device
    gamma = _uniform(batch, config.gamma_min, config.gamma_max, device, generator).view(-1, 1, 1, 1)
    contrast = _uniform(
        batch, config.contrast_min, config.contrast_max, device, generator
    ).view(-1, 1, 1, 1)
    signed_gamma = torch.sign(image) * torch.abs(image).clamp_min(1e-6).pow(gamma)
    mean = signed_gamma.mean(dim=(-2, -1), keepdim=True)
    image = mean + contrast * (signed_gamma - mean)

    height, width = image.shape[-2:]
    y = torch.linspace(-1, 1, height, device=device).view(1, 1, height, 1)
    x = torch.linspace(-1, 1, width, device=device).view(1, 1, 1, width)
    coefficients = _uniform(
        batch * 4, -config.bias_strength, config.bias_strength, device, generator
    ).view(batch, 4, 1, 1)
    field = (
        coefficients[:, 0:1] * x
        + coefficients[:, 1:2] * y
        + coefficients[:, 2:3] * x.square()
        + coefficients[:, 3:4] * y.square()
    )
    image = image * torch.exp(field)
    if config.noise_std > 0:
        noise = torch.randn(image.shape, device=device, dtype=image.dtype, generator=generator)
        image = image + noise * config.noise_std
    return image


def augment_bilateral_batch(
    image: torch.Tensor,
    labels: torch.Tensor | None,
    config: AugmentationConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not config.enabled:
        return image, labels
    image, labels = _spatial_transform(image, labels, config, generator)
    flip = torch.rand(image.shape[0], device=image.device, generator=generator) < config.lr_flip_probability
    if bool(flip.any()):
        image = image.clone()
        image[flip] = torch.flip(image[flip], dims=(-2,))
        if labels is not None:
            labels = labels.clone()
            labels[flip] = swap_bilateral_labels(torch.flip(labels[flip], dims=(-2,)))
    return _intensity_transform(image, config, generator), labels


def augment_bilateral_teacher_student(
    image: torch.Tensor,
    labels: torch.Tensor | None,
    config: AugmentationConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return strong Student and geometrically aligned weak Teacher images.

    A common anatomical transform is applied once. The Teacher sees that
    weakly augmented image, while gamma/contrast/bias/noise are applied only
    to the Student. Consequently their probability maps remain registered.
    """

    if not config.enabled:
        return image, image, labels
    weak, labels = _spatial_transform(image, labels, config, generator)
    flip = (
        torch.rand(image.shape[0], device=image.device, generator=generator)
        < config.lr_flip_probability
    )
    if bool(flip.any()):
        weak = weak.clone()
        weak[flip] = torch.flip(weak[flip], dims=(-2,))
        if labels is not None:
            labels = labels.clone()
            labels[flip] = swap_bilateral_labels(torch.flip(labels[flip], dims=(-2,)))
    strong = _intensity_transform(weak.clone(), config, generator)
    return strong, weak, labels
