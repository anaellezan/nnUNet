import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes

import SimpleITK as sitk

import os




## FAIRE LE DISTINGO ENTRE RAW DONEES SOURCES ET LE RAW DE NNUNET 
if __name__ == "__main__":
    # nnunet_dir = '/media/sharedata/myosaiq_challenge/ana_test/nnUNet_raw/raw_data'
    # source_dir = '/media/sharedata/atriumCT/data_anonymized/'
    source_dir = '/media/sharedata/atriumCT/corrected_data/'

    nnunet_dir = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/'
    # dataset_name = 'Dataset002_LA_CT00'
    # dataset_name = 'Dataset003_LA_CT00_corrected'
    dataset_name = 'Dataset004_LA_CT00_corrected_voted'
    imagestr = join(nnunet_dir, dataset_name, 'imagesTr')
    labelstr = join(nnunet_dir, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    all_cases = [f[:-4] for f in os.listdir(f'{source_dir}GTImages')]
    all_cases.sort(key=int)
    print(f'\n{len(all_cases)} cases in the source dir.\n')

   
    
    for i in range(len(all_cases)):
        # copy images
        print(f'{source_dir}GTImages/{all_cases[i]}.mha')
        print(f'{imagestr}/la_trans_corrected_{all_cases[i]}_{i:03}_0000.mha')
        shutil.copy(f'{source_dir}GTImages/{all_cases[i]}.mha', f'{imagestr}/la_trans_corrected_{all_cases[i]}_{i:03}_0000.mha')

        # copy labels
        # print(f'{source_dir}GTmasks/{all_cases[i]}.mha')
        print(f'{source_dir}GTmasks2/{all_cases[i]}.mha')
        print(f'{labelstr}/la_trans_corrected_{all_cases[i]}_{i:03}.mha')
        # shutil.copy(f'{source_dir}GTmasks/{all_cases[i]}.mha', f'{labelstr}/la_trans_corrected_{all_cases[i]}_{i:03}.mha')
        shutil.copy(f'{source_dir}GTmasks2/{all_cases[i]}.mha', f'{labelstr}/la_trans_corrected_{all_cases[i]}_{i:03}.mha')


    generate_dataset_json(output_folder=join(nnunet_dir, dataset_name),
                          channel_names={0: 'CT'},
                          labels={'background': 0, 'LA': 1},
                          num_training_cases=len(all_cases),
                          file_ending='.mha',
                          dataset_name=dataset_name,
                          )
