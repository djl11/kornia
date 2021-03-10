import ivy
from typing import List


# _se_to_mask
def _se_to_mask(se: ivy.Tensor) -> ivy.Tensor:
    se_h, se_w = se.shape
    se_flat = ivy.reshape(se, (-1,))
    num_feats = se_h * se_w
    i_s = ivy.expand_dims(ivy.arange(num_feats, dev=ivy.dev_str(se)), -1)
    y_s = i_s % se_h
    x_s = i_s // se_h
    indices = ivy.concatenate((i_s, ivy.zeros_like(i_s, dtype_str='int32'), x_s, y_s), -1)
    out = ivy.scatter_nd(
        indices, ivy.cast(se_flat >= 0, ivy.dtype_str(se)), (num_feats, 1, se_h, se_w), dev=ivy.dev_str(se))
    return out


# noinspection PyShadowingNames
def dilation(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the dilated image applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Dilated image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((3, 3))
        >>> dilated_img = dilation(tensor, kernel)
    """
    if ivy.backend == 'torch' and not isinstance(tensor, ivy.Tensor):
        raise TypeError("Input type is not an ivy.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            ivy.get_num_dims(tensor)))

    if ivy.backend == 'torch' and not isinstance(kernel, ivy.Tensor):
        raise TypeError("Kernel type is not a ivy.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            ivy.get_num_dims(kernel)))

    # prepare kernel
    se_d: ivy.Tensor = kernel - 1.
    kernel_d: ivy.Tensor = ivy.transpose(_se_to_mask(se_d), (2, 3, 1, 0))

    # pad
    se_h, se_w = kernel.shape

    output: ivy.Tensor = ivy.reshape(tensor,
                                     (tensor.shape[0] * tensor.shape[1], 1, tensor.shape[2], tensor.shape[3]))
    output = ivy.reduce_max(ivy.conv2d(output, kernel_d, 1, 'SAME', data_format='NCHW') +
                            ivy.reshape(se_d, (1, -1, 1, 1)), [1])
    return ivy.reshape(output, tensor.shape)


# erosion
# noinspection PyShadowingNames
def erosion(tensor: ivy.Tensor, kernel: ivy.Tensor) -> ivy.Tensor:
    r"""Returns the eroded image applying the same kernel in each channel.

    The kernel must have 2 dimensions, each one defined by an odd number.

    Args:
       tensor (ivy.Tensor): Image with shape :math:`(B, C, H, W)`.
       kernel (ivy.Tensor): Structuring element with shape :math:`(H, W)`.

    Returns:
       ivy.Tensor: Eroded image with shape :math:`(B, C, H, W)`.

    Example:
        >>> tensor = ivy.random_uniform(shape=(1, 3, 5, 5))
        >>> kernel = ivy.ones((5, 5))
        >>> output = erosion(tensor, kernel)
    """
    if ivy.backend == 'torch' and not isinstance(tensor, ivy.Tensor):
        raise TypeError("Input type is not an ivy.Tensor. Got {}".format(type(tensor)))

    if len(tensor.shape) != 4:
        raise ValueError("Input size must have 4 dimensions. Got {}".format(
            ivy.get_num_dims(tensor)))

    if ivy.backend == 'torch' and not isinstance(kernel, ivy.Tensor):
        raise TypeError("Kernel type is not a ivy.Tensor. Got {}".format(type(kernel)))

    if len(kernel.shape) != 2:
        raise ValueError("Kernel size must have 2 dimensions. Got {}".format(
            ivy.get_num_dims(kernel)))

    # prepare kernel
    se_e: ivy.Tensor = kernel - 1.
    kernel_e: ivy.Tensor = ivy.transpose(_se_to_mask(se_e), (2, 3, 1, 0))

    # pad
    se_h, se_w = kernel.shape
    pad_e: List[List[int]] = [[0]*2, [0]*2, [se_h // 2, se_w // 2], [se_h // 2, se_w // 2]]

    output: ivy.Tensor = ivy.reshape(tensor,
                                     (tensor.shape[0] * tensor.shape[1], 1, tensor.shape[2], tensor.shape[3]))
    output = ivy.constant_pad(output, pad_e, value=1.)
    output = ivy.reduce_min(ivy.conv2d(output, kernel_e, 1, 'VALID', data_format='NCHW') -
                            ivy.reshape(se_e, (1, -1, 1, 1)), [1])
    return ivy.reshape(output, tensor.shape)
