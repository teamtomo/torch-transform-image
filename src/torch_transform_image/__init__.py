"""Real space transformations of 2D/3D images in PyTorch"""

__version__ = '0.1.0'
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_transform_image.transforms_2d import affine_transform_image_2d
from torch_transform_image.transforms_3d import affine_transform_image_3d

__all__ = [
    'affine_transform_image_2d',
    'affine_transform_image_3d',
]
