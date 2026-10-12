from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


def _centered_coordinates(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=device, dtype=torch.float32) - height // 2
    x = torch.arange(width, device=device, dtype=torch.float32) - width // 2
    return torch.meshgrid(y, x, indexing="ij")


@lru_cache(maxsize=16)
def _cpu_band_bank(height: int, width: int, angles: int, width_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = _centered_coordinates(height, width, torch.device("cpu"))
    band_width = min(height, width) * width_fraction
    masks = []
    for angle_index in range(angles):
        angle = np.pi * angle_index / angles
        normal_distance = torch.abs(-np.sin(angle) * xx + np.cos(angle) * yy)
        masks.append(normal_distance <= band_width / 2.0)
    bank = torch.stack(masks)
    probability = bank.float().mean(dim=0).clamp_min(1.0 / angles)
    return bank, probability


class RandomBandLoss(torch.nn.Module):
    def __init__(self, angles: int = 180, width_fraction: float = 0.25) -> None:
        super().__init__()
        self.angles = angles
        self.width_fraction = width_fraction

    def bank_and_probability(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bank, probability = _cpu_band_bank(height, width, self.angles, self.width_fraction)
        return bank.to(device=device), probability.to(device=device)

    def forward(
        self,
        probability_map: torch.Tensor,
        target: torch.Tensor,
        angle_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if probability_map.shape != target.shape:
            raise ValueError("Probability map and target must have identical shapes")
        batch, _, height, width = probability_map.shape
        bank, inclusion = self.bank_and_probability(height, width, probability_map.device)
        if angle_indices is None:
            angle_indices = torch.randint(self.angles, (batch,), device=probability_map.device)
        selected = bank[angle_indices].unsqueeze(1)
        with torch.amp.autocast(probability_map.device.type, enabled=False):
            pred_fft = torch.fft.fftshift(
                torch.fft.fft2(probability_map.float(), norm="ortho"), dim=(-2, -1)
            )
            target_fft = torch.fft.fftshift(
                torch.fft.fft2(target.float(), norm="ortho"), dim=(-2, -1)
            )
            complex_l1 = torch.abs(pred_fft - target_fft)
            weighted = selected * complex_l1 / inclusion.view(1, 1, height, width)
            return weighted.sum(dim=(-2, -1)).mean() / (height * width)

    def full_spectrum_loss(self, probability_map: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(probability_map.device.type, enabled=False):
            difference = torch.abs(
                torch.fft.fft2(probability_map.float(), norm="ortho")
                - torch.fft.fft2(target.float(), norm="ortho")
            )
            return difference.mean()


def _partner_source(height: int, width: int, device: torch.device) -> torch.Tensor:
    row = torch.arange(height, device=device)[:, None].expand(height, width)
    column = torch.arange(width, device=device)[None, :].expand(height, width)
    flat = row * width + column
    partner = ((-row) % height) * width + ((-column) % width)
    return torch.minimum(flat, partner).reshape(-1)


def is_hermitian_mask(mask: torch.Tensor) -> bool:
    height, width = mask.shape[-2:]
    row = torch.arange(height, device=mask.device)
    column = torch.arange(width, device=mask.device)
    partner = mask[..., (-row) % height, :][..., :, (-column) % width]
    return bool(torch.equal(mask, partner))


def _variable_density_probability(
    height: int,
    width: int,
    center_fraction: float,
    base_keep_rate: float,
    radial_power: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fy = torch.fft.fftfreq(height, device=device).abs()[:, None]
    fx = torch.fft.fftfreq(width, device=device).abs()[None, :]
    radius = torch.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2).clamp(0, 1)
    center = radius <= center_fraction
    score = (1.0 - radius).pow(radial_power).clamp_min(1e-6)
    periphery = ~center
    low, high = 0.0, 1e6
    for _ in range(50):
        scale = (low + high) / 2.0
        mean = torch.clamp(score[periphery] * scale, max=1.0).mean().item()
        if mean < base_keep_rate:
            low = scale
        else:
            high = scale
    probability = torch.clamp(score * ((low + high) / 2.0), max=1.0)
    probability[center] = 1.0
    return probability, center


def complementary_fourier_masks(
    batch_size: int,
    height: int,
    width: int,
    center_fraction: float = 0.08,
    base_keep_rate: float = 0.70,
    radial_power: int = 4,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    probability, center = _variable_density_probability(
        height, width, center_fraction, base_keep_rate, radial_power, device
    )
    source = _partner_source(height, width, device)
    probability_flat = probability.reshape(-1)
    base_random = torch.rand((batch_size, height * width), device=device, generator=generator)
    split_random = torch.rand((batch_size, height * width), device=device, generator=generator)
    symmetric_base_random = base_random.gather(1, source.unsqueeze(0).expand(batch_size, -1))
    symmetric_split_random = split_random.gather(1, source.unsqueeze(0).expand(batch_size, -1))
    base = symmetric_base_random < probability_flat.unsqueeze(0)
    choose_theta = symmetric_split_random < 0.5
    theta = (base & choose_theta).reshape(batch_size, height, width)
    lam = (base & ~choose_theta).reshape(batch_size, height, width)
    theta[:, center] = True
    lam[:, center] = True
    return theta.unsqueeze(1), lam.unsqueeze(1), center


def fourier_view(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and image.shape[1] != 1:
        mask = mask.expand(-1, image.shape[1], -1, -1)
    with torch.amp.autocast(image.device.type, enabled=False):
        spectrum = torch.fft.fft2(image.float(), norm="ortho")
        reconstructed = torch.fft.ifft2(spectrum * mask, norm="ortho")
        return torch.abs(reconstructed)


def make_complementary_views(
    image: torch.Tensor,
    center_fraction: float,
    base_keep_rate: float,
    radial_power: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, height, width = image.shape
    theta, lam, _ = complementary_fourier_masks(
        batch,
        height,
        width,
        center_fraction,
        base_keep_rate,
        radial_power,
        image.device,
    )
    return fourier_view(image, theta), fourier_view(image, lam), theta, lam
