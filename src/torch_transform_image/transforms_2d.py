from typing import Literal, Optional

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
        output_shape: Optional[tuple] = None,
        yx_matrices: bool = False,
) -> torch.Tensor:
    # grab image dimensions
    if output_shape:
        h, w = output_shape
    else:
        h, w = image.shape[-2:]

    if not yx_matrices:
        matrices[..., :2, :2] = (
            torch.flip(matrices[..., :2, :2], dims=(-2, -1))
        )
        matrices[..., :2, 2] = torch.flip(matrices[..., :2, 2], dims=(-1,))

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(h, w), device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (h, w, yxw)
    grid = einops.rearrange(grid, 'h w yxw -> h w yxw 1')
    grid = matrices @ grid
    grid = grid[..., :2, 0]  # dehomogenise coordinates: (..., h, w, yxw, 1) -> (..., h, w, yx)

    # sample image at transformed positions
    result = sample_image_2d(image, coordinates=grid, interpolation=interpolation)
    return result


def rotate_shift_image_2d(
    image: torch.Tensor,
    angle: int | float = 0,
    shift: list[float | int] | tuple[float | int, float | int] = (0, 0),
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
    rotate_first: bool = True,
) -> torch.Tensor:
    """This is a wrapper function for easy 2D shifts and rotations.

    Rotations are specified in degrees and performed CCW around the
    center of the image. Rotating and shifting can be performed in
    either order. Currently, only a single shift and a single rotation
    are allowed.

    Parameters
    ----------
    image : torch.Tensor
        The image to be shifted/rotated.
    angle : int | float, optional
        The angle in degrees by which to rotate the image.
    shift : list[float | int] | tuple[float | int, float | int],
    optional
        The number of pixels by which to shift the image. Must be a list
        or tuple of length 2 in the form (y, x).
    interpolation : Literal["nearest", "bilinear", "bicubic"], optional
        The interpolation method to use. Default is "bicubic".
    rotate_first : bool, optional
        If True, the image is rotated first and then shifted. If False,
        the image is shifted first and then rotated. Default is True.

    Returns
    -------
    torch.Tensor
        The shifted and/or rotated image.
    """
    image_center = 0
    if angle:
        h, w = image.shape[-2:]
        image_center = dft_center(
            image_shape=(h, w), device=image.device, fftshift=True, rfft=False
            )

    center_tensor = torch.as_tensor(image_center, device=image.device, dtype=torch.float32)
    angle_tensor = torch.as_tensor(angle, device=image.device, dtype=torch.float32)
    shift_tensor = torch.as_tensor(shift, device=image.device, dtype=torch.float32)

    if angle_tensor.numel() > 1 or shift_tensor.numel() > 2:
        raise NotImplementedError(
            "Only single angle and single shift values are supported."
            )

    if rotate_first:
        matrix = (
            T(center_tensor) @
            R(angle_tensor) @
            T(shift_tensor) @
            T(-center_tensor)
        )
    else:
        matrix = (
            T(center_tensor) @
            T(shift_tensor) @
            R(angle_tensor) @
            T(-center_tensor)
        )

    return affine_transform_image_2d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        yx_matrices=True,
    )
