import ivy
from kornia.morphology.basic_operators import dilation, erosion


# open
# noinspection PyShadowingBuiltins,PyShadowingNames
def open(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the opened image, (that means, erosion after a dilation) applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((3, 3))
        >>> opened_img = open(tensor, kernel)
    """

    if ivy.backend == 'torch' and not isinstance(tensor, ivy.Tensor):
        raise TypeError("Input type is not a ivy.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            ivy.get_num_dims(tensor)))

    if ivy.backend == 'torch' and not isinstance(kernel, ivy.Tensor):
        raise TypeError("Kernel type is not a ivy.Tensor. Got {}".format(
            type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            ivy.get_num_dims(kernel)))

    return dilation(erosion(tensor, kernel), kernel)


# close
# noinspection PyShadowingNames
def close(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the closed image, (that means, dilation after an erosion) applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((3, 3))
        >>> closed_img = close(tensor, kernel)
    """

    if ivy.backend == 'torch' and not isinstance(tensor, ivy.Tensor):
        raise TypeError("Input type is not a ivy.Tensor. Got {}".format(
            type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            ivy.get_num_dims(tensor)))

    if ivy.backend == 'torch' and not isinstance(kernel, ivy.Tensor):
        raise TypeError("Kernel type is not a ivy.Tensor. Got {}".format(
            type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            ivy.get_num_dims(kernel)))

    return erosion(dilation(tensor, kernel), kernel)
