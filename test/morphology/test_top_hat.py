import ivy
import pytest
import numpy as np
from ivy_tests import helpers
# noinspection PyProtectedMember
from torch.autograd import gradcheck
from kornia.morphology.morphology import top_hat
# noinspection PyProtectedMember
from kornia.morphology.basic_operators import _se_to_mask


class TestTopHat:

    def test_smoke(self, dev_str, dtype_str, call):
        kernel = ivy.cast(ivy.random_uniform(shape=(3, 3), dev=dev_str), dtype_str)
        assert call(_se_to_mask, kernel) is not None

    @pytest.mark.parametrize(
        "shape", [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 5, 5)])
    @pytest.mark.parametrize(
        "kernel", [(3, 3), (5, 5)])
    def test_cardinality(self, dev_str, dtype_str, call, shape, kernel):
        img = ivy.ones(shape, dtype_str=dtype_str, dev=dev_str)
        krnl = ivy.ones(kernel, dtype_str=dtype_str, dev=dev_str)
        assert top_hat(img, krnl).shape == shape

    def test_value(self, dev_str, dtype_str, call):
        input_ = ivy.array([[0.5, 1., 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]],
                           dev=dev_str, dtype_str=dtype_str)[None, None, :, :]
        kernel = ivy.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dev=dev_str, dtype_str=dtype_str)
        expected = np.array([[0., 0.5, 0.], [0.2, 0., 0.5], [0., 0.5, 0.]])[None, None, :, :]
        assert np.allclose(call(top_hat, input_, kernel), expected, atol=1e-7)

    def test_exception(self, dev_str, dtype_str, call):
        input_ = ivy.ones((1, 1, 3, 4), dev=dev_str, dtype_str=dtype_str)
        kernel = ivy.ones((3, 3), dev=dev_str, dtype_str=dtype_str)

        with pytest.raises(ValueError):
            test = ivy.ones((2, 3, 4), dev=dev_str, dtype_str=dtype_str)
            assert top_hat(test, kernel)

        with pytest.raises(ValueError):
            test = ivy.ones((2, 3, 4), dev=dev_str, dtype_str=dtype_str)
            assert top_hat(input_, test)

        if call is not helpers.torch_call:
            return

        with pytest.raises(TypeError):
            assert top_hat([0.], kernel)

        with pytest.raises(TypeError):
            assert top_hat(input_, [0.])

    @pytest.mark.grad
    def test_gradcheck(self, dev_str, dtype_str, call):
        if call is not helpers.torch_call:
            # ivy gradcheck method not yet implemented
            pytest.skip()
        input_ = ivy.variable(ivy.cast(ivy.random_uniform(shape=(2, 3, 4, 4), dev=dev_str), 'float64'))
        kernel = ivy.variable(ivy.cast(ivy.random_uniform(shape=(3, 3), dev=dev_str), 'float64'))
        assert gradcheck(top_hat, (input_, kernel), raise_exception=True)

    @pytest.mark.jit
    def test_jit(self, dev_str, dtype_str, call):
        op = top_hat
        if call in [helpers.jnp_call, helpers.torch_call]:
            # compiled jax tensors do not have device_buffer attribute, preventing device info retrieval,
            # pytorch scripting does not support .type() casting, nor Union or Numbers for type hinting
            pytest.skip()
        op_compiled = ivy.compile_fn(op)

        input_ = ivy.cast(ivy.random_uniform(shape=(1, 2, 7, 7), dev=dev_str), dtype_str)
        kernel = ivy.ones((3, 3), dev=dev_str, dtype_str=dtype_str)

        actual = call(op_compiled, input_, kernel)
        expected = call(op, input_, kernel)

        assert np.allclose(actual, expected)
