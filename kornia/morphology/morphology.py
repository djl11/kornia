import ivy
from kornia.morphology.open_close import open, close
from kornia.morphology.basic_operators import dilation, erosion


# morphological gradient
# noinspection PyShadowingNames,PyUnresolvedReferences
def gradient(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the morphological gradient of an image.

    That means, (dilation - erosion) applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((3, 3))
        >>> gradient_img = gradient(tensor, kernel)
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

    return dilation(tensor, kernel) - erosion(tensor, kernel)


# top_hat
# noinspection PyShadowingNames,PyUnresolvedReferences
def top_hat(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the top hat tranformation of an image.

    That means, (image - opened_image) applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    See :class:`~kornia.morphology.open` for details.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((3, 3))
        >>> top_hat_img = top_hat(tensor, kernel)
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

    return tensor - open(tensor, kernel)


# black_hat
# noinspection PyShadowingNames,PyUnresolvedReferences
def black_hat(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the black hat tranformation of an image.

    That means, (closed_image - image) applying the same kernel in each channel.
    The kernel must have 2 dimensions, each one defined by an odd number.

    See :class:`~kornia.morphology.close` for details.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Top hat transformated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((3, 3))
        >>> black_hat_img = black_hat(tensor, kernel)
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

    return close(tensor, kernel) - tensor
