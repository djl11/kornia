import ivy
import pytest
import numpy as np
from ivy_tests import helpers
# noinspection PyProtectedMember
from torch.autograd import gradcheck
from kornia.morphology.open_close import open
# noinspection PyProtectedMember
from kornia.morphology.basic_operators import _se_to_mask


class TestOpen:

    def test_smoke(self, dev_str, dtype_str, call):
        kernel = ivy.cast(ivy.random_uniform(shape=(3, 3), dev_str=dev_str), dtype_str)
        assert call(_se_to_mask, kernel) is not None

    @pytest.mark.parametrize(
        "shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 5, 5)])
    @pytest.mark.parametrize(
        "kernel", [(3, 3), (5, 5)])
    def test_cardinality(self, dev_str, dtype_str, call, shape, kernel):
        img = ivy.ones(shape, dtype_str=dtype_str, dev_str=dev_str)
        krnl = ivy.ones(kernel, dtype_str=dtype_str, dev_str=dev_str)
        assert open(img, krnl).shape == shape

    def test_value(self, dev_str, dtype_str, call):
        input_ = ivy.array([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                           dev_str=dev_str, dtype_str=dtype_str)[None, None, :, :]
        kernel = ivy.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dev_str=dev_str, dtype_str=dtype_str)
        expected = np.array([[0.5, 0.5, 0.3], [0.5, 0.3, 0.3], [0.4, 0.4, 0.2]])[None, None, :, :]
        assert np.allclose(call(open, input_, kernel), expected)

    def test_exception(self, dev_str, dtype_str, call):
        input_ = ivy.ones((1, 1, 3, 4), dev_str=dev_str, dtype_str=dtype_str)
        kernel = ivy.ones((3, 3), dev_str=dev_str, dtype_str=dtype_str)

        with pytest.raises(ValueError):
            test = ivy.ones((2, 3, 4), dev_str=dev_str, dtype_str=dtype_str)
            assert open(test, kernel)

        with pytest.raises(ValueError):
            test = ivy.ones((2, 3, 4), dev_str=dev_str, dtype_str=dtype_str)
            assert open(input_, test)

        if call is not helpers.torch_call:
            return

        with pytest.raises(TypeError):
            assert open([0.], kernel)

        with pytest.raises(TypeError):
            assert open(input_, [0.])

    @pytest.mark.grad
    def test_gradcheck(self, dev_str, dtype_str, call):
        if call is not helpers.torch_call:
            # ivy gradcheck method not yet implemented
            pytest.skip()
        input_ = ivy.variable(ivy.cast(ivy.random_uniform(shape=(2, 3, 4, 4), dev_str=dev_str), 'float64'))
        kernel = ivy.variable(ivy.cast(ivy.random_uniform(shape=(3, 3), dev_str=dev_str), 'float64'))
        assert gradcheck(open, (input_, kernel), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, dev_str, dtype_str, call):
        op = open
        if call in [helpers.jnp_call, helpers.torch_call]:
            # compiled jax tensors do not have device_buffer attribute, preventing device info retrieval,
            # pytorch scripting does not support .type() casting, nor Union or Numbers for type hinting
            pytest.skip()
        op_compiled = ivy.compile_fn(op)

        input_ = ivy.cast(ivy.random_uniform(shape=(1, 2, 7, 7), dev_str=dev_str), dtype_str)
        kernel = ivy.ones((3, 3), dev_str=dev_str, dtype_str=dtype_str)

        actual = call(op_compiled, input_, kernel)
        expected = call(op, input_, kernel)

        assert np.allclose(actual, expected)
