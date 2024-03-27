import nnunetv2.atriumCT_model.preprocessing.utils as utils
import numpy as np

def test_bidon():
    arr = np.random.rand(3,2,3)
    utils.test_empty_arr(arr, "path", limit_percentage=5)