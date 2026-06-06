"""Paired 3D data augmentation for multi-region DCE-MRI.

Same-timepoint pairs (Breast + ALN) receive identical spatial
augmentations to preserve anatomical relationships.

Supports multi-channel volumes (4, D, H, W) from DCE kinetic maps.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate, shift


class PairedTransform3D:
    """Apply identical spatial augmentations to a pair of 3D volumes.

    Volumes can be single-channel (D, H, W) or multi-channel (C, D, H, W).
    Spatial transforms are applied identically to both volumes;
    intensity augmentation is applied independently.
    """

    def __init__(
        self,
        rotation_range: float = 10.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        intensity_shift: float = 0.1,
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
    ) -> None:
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.intensity_shift = intensity_shift
        self.noise_std = noise_std
        self.flip_prob = flip_prob

    def __call__(
        self, vol_a: np.ndarray, vol_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply paired augmentation.

        Parameters
        ----------
        vol_a, vol_b : (C, D, H, W) or (D, H, W)
            Multi-channel or single-channel 3D volumes.

        Returns
        -------
        Augmented copies of vol_a, vol_b with same shapes.
        """
        rng = np.random.default_rng()

        # Work with copies to avoid modifying originals
        vol_a = vol_a.copy()
        vol_b = vol_b.copy()

        # Determine if multi-channel
        multi_ch = vol_a.ndim == 4

        # Random rotation (same angle/axes for both volumes)
        angle = rng.uniform(-self.rotation_range, self.rotation_range)
        # spatial axes: for (C,D,H,W) use (1,2), (1,3), (2,3); for (D,H,W) use (0,1), (0,2), (1,2)
        if multi_ch:
            axes_choices = [(1, 2), (1, 3), (2, 3)]
        else:
            axes_choices = [(0, 1), (0, 2), (1, 2)]
        axes = axes_choices[rng.integers(0, 3)]
        vol_a = rotate(vol_a, angle, axes=axes, reshape=False, order=1)
        vol_b = rotate(vol_b, angle, axes=axes, reshape=False, order=1)

        # Random flip (same for both)
        if rng.random() < self.flip_prob:
            if multi_ch:
                axis = rng.integers(1, 4)  # flip spatial dims only
            else:
                axis = rng.integers(0, 3)
            vol_a = np.flip(vol_a, axis=axis).copy()
            vol_b = np.flip(vol_b, axis=axis).copy()

        # Random translation (same for both)
        if multi_ch:
            shift_val = np.concatenate([[0], rng.uniform(-2, 2, size=3)])
        else:
            shift_val = rng.uniform(-2, 2, size=3)
        vol_a = shift(vol_a, shift_val, order=1)
        vol_b = shift(vol_b, shift_val, order=1)

        # Intensity augmentation (independent per volume)
        for vol in [vol_a, vol_b]:
            vol += rng.uniform(-self.intensity_shift, self.intensity_shift)
            vol += rng.normal(0, self.noise_std, size=vol.shape).astype(np.float32)

        return vol_a, vol_b
