from typing import Literal, Optional

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T
from torch_grid_utils import coordinate_grid, dft_center
from torch_image_interpolation import sample_image_3d


def affine_transform_image_3d(
        image: torch.Tensor,
        matrices: torch.Tensor,
        interpolation: Literal['nearest', 'trilinear'],
        output_shape: Optional[tuple] = None,
        zyx_matrices: bool = False,
) -> torch.Tensor:
    # grab image dimensions
    if output_shape:
        d, h, w = output_shape
    else:
        d, h, w = image.shape[-3:]

    if not zyx_matrices:
        matrices[..., :3, :3] = torch.flip(matrices[..., :3, :3], dims=(-2, -1))
        matrices[..., :3, 3] = torch.flip(matrices[..., :3, 3], dims=(-1,))

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(d, h, w), device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (d, h, w, zyxw)
    grid = einops.rearrange(grid, 'd h w zyxw -> d h w zyxw 1')
    grid = matrices @ grid
    grid = grid[..., :3, 0]  # dehomogenise coordinates: (..., d, h, w, zyxw, 1) -> (..., d, h, w, zyx)

    # sample image at transformed positions
    result = sample_image_3d(image, coordinates=grid, interpolation=interpolation)
    return result


def rotate_then_shift_image_3d(
    image: torch.Tensor,
    rotate_zyx: list[float] | tuple[float, float, float] = (0, 0, 0),
    shift_zyx: list[float] | tuple[float, float] = (0, 0),
    interpolation: Literal["nearest", "trilinear"] = "trilinear",
) -> torch.Tensor:
    """
    Modify from 2D function docsrings
    """
    image_center = 0
    if any(rotate_zyx):
        d, h, w = image.shape[-3:]
        image_center = dft_center(
            image_shape=(d, h, w), device=image.device, fftshift=True, rfft=False
            )

    center_tensor = torch.as_tensor(image_center, device=image.device, dtype=torch.float32)
    rotate_tensor = torch.as_tensor(rotate_zyx, device=image.device, dtype=torch.float32)
    # Because shift is applied to the coordinate grid, it must be
    # negated to produce a positive (up/right) shift on the image.
    shift_tensor = -torch.as_tensor(shift_zyx, device=image.device, dtype=torch.float32)

    if (num_angles := rotate_tensor.numel()) != 3:
        raise ValueError(
            f"3 angles are required but {num_angles} were supplied: {rotate_zyx}."
        )
    if (num_shifts := shift_tensor.numel()) != 3:
        raise ValueError(
            f"3 shifts are required but {num_shifts} were supplied: {shift_zyx}."
        )

    matrix = (
        T(center_tensor) @
        Rx(rotate_tensor[0]) @
        Ry(rotate_tensor[1]) @
        Rz(rotate_tensor[2]) @
        T(shift_tensor) @
        T(-center_tensor)
    )

    return affine_transform_image_3d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        zyx_matrices=True,
    )
