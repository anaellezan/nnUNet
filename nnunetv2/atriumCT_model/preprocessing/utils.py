import numpy as np


def get_intersection(mask_1, mask_2):
    return (mask_1 & mask_2)


def get_union(mask_1, mask_2):
    return (mask_1 | mask_2)


def test_empty_arr(arr, im_path, limit_percentage=5):
    filled_percentage = (
        np.sum((arr != 0)) / (arr.shape[0] * arr.shape[1] * arr.shape[2]) * 100
    )
    if filled_percentage < limit_percentage:
        print(
            f'WARNING: only {filled_percentage} % filled in the image:\n{im_path}\n'
        )


def test_unique_val(arr, im_path, nb_unique_values_expected):
    nb_unique_values = len(np.unique(arr))
    print(np.unique(arr))
    print(im_path)
    if nb_unique_values != nb_unique_values_expected:
        print(
            f'WARNING: only {nb_unique_values} unique values, {nb_unique_values_expected} expected.\n{im_path}\n'
        )
