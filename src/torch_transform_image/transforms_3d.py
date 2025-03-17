from typing import Literal

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_3d


def affine_transform_image_3d(
    image: torch.Tensor,
    matrices: torch.Tensor,
    interpolation: Literal['nearest', 'trilinear']
) -> torch.Tensor:
    # grab image dimensions
    d, h, w = image.shape[-3:]

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
