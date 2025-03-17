from typing import Literal

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d


def affine_transform_image_2d(
    image: torch.Tensor,
    matrices: torch.Tensor,
    interpolation: Literal['nearest', 'bilinear', 'bicubic']
) -> torch.Tensor:
    # grab image dimensions
    h, w = image.shape[-2:]

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(h, w), device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (h, w, yxw)
    grid = einops.rearrange(grid, 'h w yxw -> h w yxw 1')
    grid = matrices @ grid
    grid = grid[..., :2, 0]  # dehomogenise coordinates: (h, w, yxw, 1) -> (h, w, yx)

    # sample image at transformed positions
    result = sample_image_2d(image, coordinates=grid, interpolation=interpolation)
    return result
