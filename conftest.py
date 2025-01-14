import pytest
import ivy.jax
import ivy.tensorflow
import ivy.torch
import ivy.numpy
from typing import List, Dict
import itertools
from ivy_tests import helpers
helpers.exclude(['mxnd', 'mxsym'])


FW_STRS = ['numpy', 'jax', 'tensorflow', 'tensorflow_graph', 'torch']


def get_test_devices() -> Dict[ivy.Framework, List[str]]:
    """Creates a list of device strings to test the source code as values.
    CUDA devices will be test only in case the current hardware supports it.

    Return:
        dict(ivy.Framework, list(device_str)): list with devices names.
    """
    devices_dict: Dict[str, List[str]] = dict()
    for fw_str, (f, _) in zip(FW_STRS, helpers.f_n_calls()):
        new_list = list()
        new_list.append('cpu:0')
        if f.gpu_is_available():
            new_list.append('cuda:0')
        if f.tpu_is_available():
            new_list.append('tpu:0')
        devices_dict[fw_str] = new_list
    return devices_dict


# setup the global containers to test the source code
TEST_DEV_STRS: Dict[ivy.Framework, List[str]] = get_test_devices()
TEST_DTYPE_STRS: List[str] = ['float16', 'float32', 'float64']
TEST_FRAMEWORKS: Dict[str, ivy.Framework] = {'numpy': ivy.numpy,
                                             'jax': ivy.jax,
                                             'tensorflow': ivy.tensorflow,
                                             'tensorflow_graph': ivy.tensorflow,
                                             'torch': ivy.torch}
TEST_CALL_METHODS: Dict[str, callable] = {'numpy': helpers.np_call,
                                          'jax': helpers.jnp_call,
                                          'tensorflow': helpers.tf_call,
                                          'tensorflow_graph': helpers.tf_graph_call,
                                          'torch': helpers.torch_call}


@pytest.fixture(autouse=True)
def run_around_tests(f):
    with f.use:
        yield


def pytest_generate_tests(metafunc):

    dev_strs = None
    dtype_strs = None
    f_strs = None

    if 'dev_str' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dev_str')
        if raw_value == 'all':
            dev_strs = TEST_DEV_STRS
        else:
            dev_strs = raw_value.split(',')

    if 'dtype_str' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--dtype_str')
        if raw_value == 'all':
            dtype_strs = TEST_DTYPE_STRS
        else:
            dtype_strs = raw_value.split(',')

    if 'f' in metafunc.fixturenames:
        raw_value = metafunc.config.getoption('--backend')
        if raw_value == 'all':
            f_strs = TEST_FRAMEWORKS.keys()
        else:
            f_strs = raw_value.split(',')

    if dev_strs is not None and dtype_strs is not None and f_strs is not None:
        dev_n_f_n_call_tuples = list(itertools.chain.from_iterable(
            [[(item, TEST_FRAMEWORKS[f_str], TEST_CALL_METHODS[f_str]) for item in TEST_DEV_STRS[f_str]]
             for f_str in f_strs]))
        params = [(combo[1][0],) + (combo[0],) + combo[1][1:]
                  for combo in itertools.product(dtype_strs, dev_n_f_n_call_tuples)]
        metafunc.parametrize('dev_str,dtype_str,f,call', params)

    # ToDo: add full support for partial arguments later
    elif dev_strs is not None:
        metafunc.parametrize('dev_str', dev_strs)
    elif dtype_strs is not None:
        metafunc.parametrize('dtype_str', dtype_strs)


def pytest_addoption(parser):
    parser.addoption('--dev_str', action="store", default="all")
    parser.addoption('--dtype_str', action="store", default="float32")
    parser.addoption('--backend', action="store", default="all")
