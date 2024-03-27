# import pytest
import os
import argparse
import nnunetv2.atriumCT_model.preprocessing.utils as utils
import SimpleITK as sitk
from contextlib import redirect_stdout
from tqdm import tqdm


def get_files_list(path):
    files = [f for f in os.listdir( path ) if f[-4:]=='.mha']
    return files

def test_masks_integrity(path):
    files = get_files_list(path)
    print('Checking empty array with limit_percentage=8.\n')
    print(f'Checking number of unique values=2.\n')
    for file in tqdm(files):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(f'{path}/{file}'))
        utils.test_empty_arr(arr, f'{path}/{file}', limit_percentage=8)
        utils.test_unique_val(arr, f'{path}/{file}', nb_unique_values_expected=2)


def test_labels_integrity(path, nb_unique_values_expected):
    files = get_files_list(path)
    print('Checking empty array with limit_percentage=2.\n')
    print(f'Checking number of unique values={nb_unique_values_expected}.\n')
    for file in tqdm(files):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(f'{path}/{file}'))
        utils.test_empty_arr(arr, f'{path}/{file}', limit_percentage=2)
        utils.test_unique_val(arr, f'{path}/{file}', nb_unique_values_expected=nb_unique_values_expected)


def test_images_integrity(path):
    files = get_files_list(path)
    print('Checking empty array with limit_percentage=75.\n')
    for file in tqdm(files):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(f'{path}/{file}'))
        utils.test_empty_arr(arr, f'{path}/{file}', limit_percentage=75)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Give the path to check.")
    parser.add_argument("--type", type=str, required=True,
                        help="Give the of data to check among: `mask`, `label`, `image`.")
    parser.add_argument("--nb_values", type=int,
                        help="Give the number of unique values expected.")
    args = parser.parse_args()

    with open(f'{args.path}/stdout_test.txt', 'w') as f:
        with redirect_stdout(f):
            print(f'It now prints to `{args.path}/stdout_test.txt`')
            print('Checking all mha files.\n')
            if args.type == 'mask':
                test_masks_integrity(args.path)
            elif args.type == 'label':
                test_labels_integrity(args.path, args.nb_values)
            elif args.type == 'image':
                test_images_integrity(args.path)