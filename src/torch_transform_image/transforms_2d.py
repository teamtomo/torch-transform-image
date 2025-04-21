from typing import Literal

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_2d import R, T
from torch_grid_utils import coordinate_grid, dft_center
from torch_image_interpolation import sample_image_2d


def affine_transform_image_2d(
        image: torch.Tensor,
        matrices: torch.Tensor,
        interpolation: Literal['nearest', 'bilinear', 'bicubic'],
        image_center: torch.Tensor | tuple[float, ...] | None = None,
        yx_matrices: bool = False,
) -> torch.Tensor:
    # grab image dimensions
    h, w = image.shape[-2:]

    if not yx_matrices:
        matrices[..., :2, :2] = (
            torch.flip(matrices[..., :2, :2], dims=(-2, -1))
        )
        matrices[..., :2, 2] = torch.flip(matrices[..., :2, 2], dims=(-1,))

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(h, w), center=image_center, device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (h, w, yxw)
    grid = einops.rearrange(grid, 'h w yxw -> h w yxw 1')
    grid = matrices @ grid
    grid = grid[..., :2, 0]  # dehomogenise coordinates: (..., h, w, yxw, 1) -> (..., h, w, yx)

    # shift image if correcting for center
    if image_center is not None:
        grid += image_center

    # sample image at transformed positions
    result = sample_image_2d(image, coordinates=grid, interpolation=interpolation)
    return result


def shift_rotate_image_2d(
    image: torch.Tensor,
    angle: torch.Tensor | int | float = 0,
    shift: torch.Tensor | list[float | int] | int = 0,
    interpolation_mode: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
    rotate_first: bool = True,
) -> torch.Tensor:
    """This is a wrapper function to simplify 2D rotation."""
    image_center = None
    if angle:
        h, w = image.shape[-2:]
        image_center = dft_center(
            image_shape=(h, w), device=image.device, fftshift=True, rfft=False
            )

    angle_tensor = torch.as_tensor(angle, device=image.device, dtype=torch.float32)
    shift_tensor = torch.as_tensor(shift, device=image.device, dtype=torch.float32)

    if angle_tensor.numel() > 1 or shift_tensor.numel() > 2:
        raise NotImplementedError(
            "Only single angle and single shift values are supported."
            )

    if rotate_first:
        matrices = T(shift_tensor) @ R(angle_tensor)
    else:
        matrices = R(angle_tensor) @ T(shift_tensor)
    return affine_transform_image_2d(
        image=image,
        matrices=matrices,
        image_center=image_center,
        interpolation=interpolation_mode,
        yx_matrices=True,
    )
