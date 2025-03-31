import torch
from torch_transform_image import affine_transform_image_2d, affine_transform_image_3d
from torch_affine_utils.transforms_2d import T as T_2d
from torch_affine_utils.transforms_3d import T as T_3d


def test_affine_transform_image_2d():
    # set up test image with dot at (18, 14)
    image = torch.zeros((28, 28), dtype=torch.float32)
    image[18, 14] = 1
    image = image.float()

    # check that image is zero at center
    assert image[14, 14] == 0

    # define transform
    M = T_2d([4, 0])  # move coordinates up 4 in h dim

    # sample
    result = affine_transform_image_2d(
        image, M, interpolation='bicubic', yx_matrices=True,
    )

    # sanity check, array center which was 4 voxels below the dot should now be 1
    assert result.shape == image.shape
    assert result[14, 14] == 1
    assert result[18, 14] == 0


def test_affine_transform_image_3d():
    # set up test image with dot at (18, 14)
    image = torch.zeros((28, 28, 28), dtype=torch.float32)
    image[18, 14, 14] = 1
    image = image.float()

    # check that image is zero at center
    assert image[14, 14, 14] == 0

    # define transform
    M = T_3d([4, 0, 0])  # move coordinates up 4 in d dim

    # sample
    result = affine_transform_image_3d(
        image, M, interpolation='trilinear', zyx_matrices=True,
    )

    # sanity check, array center which was 4 voxels below the dot should now be 1
    assert result.shape == image.shape
    assert result[14, 14, 14] == 1
    assert result[18, 14, 14] == 0
