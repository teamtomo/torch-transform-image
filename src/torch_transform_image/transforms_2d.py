from typing import Literal

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_2d import R, T
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d


def affine_transform_image_2d(
    image: torch.Tensor,
    matrices: torch.Tensor,
    interpolation: Literal["nearest", "bilinear", "bicubic"],
    yx_matrices: bool = False,
) -> torch.Tensor:
    # grab image dimensions
    h, w = image.shape[-2:]

    if not yx_matrices:
        matrices[..., :2, :2] = torch.flip(matrices[..., :2, :2], dims=(-2, -1))
        matrices[..., :2, 2] = torch.flip(matrices[..., :2, 2], dims=(-1,))

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(h, w), device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (h, w, yxw)
    grid = einops.rearrange(grid, "h w yxw -> h w yxw 1")
    grid = matrices @ grid
    grid = grid[
        ..., :2, 0
    ]  # dehomogenise coordinates: (..., h, w, yxw, 1) -> (..., h, w, yx)

    # sample image at transformed positions
    result = sample_image_2d(image, coordinates=grid, interpolation=interpolation)
    return result


def shift_rotate_image_2d(
    image: torch.Tensor,
    angles: torch.Tensor | list[int | float] | int | float = 0,
    shifts: torch.Tensor | list[int | float] | int | float = 0,
    interpolation_mode: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
    rotate_first: bool = True,
) -> torch.Tensor:
    """This is a wrapper function to simplify 2D rotation."""
    angle_tensor = torch.as_tensor(angles, device=image.device, dtype=torch.float32)
    shift_tensor = torch.as_tensor(shifts, device=image.device, dtype=torch.float32)
    if rotate_first:
        matrices = T(shift_tensor) @ R(angle_tensor)
    else:
        matrices = R(angle_tensor) @ T(shift_tensor)
    return affine_transform_image_2d(
        image=image,
        matrices=matrices,
        interpolation=interpolation_mode,
    )
