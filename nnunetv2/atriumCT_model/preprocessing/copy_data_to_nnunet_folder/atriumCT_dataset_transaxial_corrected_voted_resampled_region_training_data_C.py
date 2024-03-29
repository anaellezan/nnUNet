''' LAST VERSION
    Building Dataset005_LA_CT00_corrected_voted_region from images and labels for a region training.

    Labels :
# 0: Hors de l'Atrium
# 1: Corps
# 2: LAA (Auricule)
# 3: RSPV (Veines)
# 4: RIPV (Veines)
# 5: LIPV (Veines)
# 6: LSPV (Veines)
# 7: opacification_defect (calcul à la main = différence entre prédiction D2 et label auricule)

    Region training labels:
# 'background': 0,
# 'whole_atrium': [1,2,3,4,5,6,7],
# 'corps_and_auricle':[1,2,7],
# 'auricle':[2,7],
# 'opacification_defect':[7]
'''

import os
import shutil

import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
)
from nnunetv2.dataset_conversion.generate_dataset_json import (
    generate_dataset_json,
)
from tqdm import tqdm

## FAIRE LE DISTINGO ENTRE RAW DONEES SOURCES ET LE RAW DE NNUNET

if __name__ == "__main__":

    source_dir = '/media/sharedata/atriumCT/corrected_data/'
    nnunet_dir = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/'

    dataset_name = 'Dataset005_LA_CT00_corrected_voted_region'

    imagestr = join(nnunet_dir, dataset_name, 'imagesTr')
    labelstr = join(nnunet_dir, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    all_cases = [f[:-4] for f in os.listdir(f'{source_dir}GTImages')]
    all_cases.sort(key=int)
    print(f'\n{len(all_cases)} cases in the source dir.\n')

    for i in tqdm(range(len(all_cases))):

        # copy images
        print(f'{source_dir}GTImages/{all_cases[i]}.mha')
        print(f'{imagestr}/la_trans_corrected_{all_cases[i]}_{i:03}_0000.mha')
        shutil.copy(
            f'{source_dir}GTImages/{all_cases[i]}.mha',
            f'{imagestr}/la_lables_opa_defect_{all_cases[i]}_{i:03}_0000.mha',
        )

        # copy labels
        print(f'{source_dir}GTlabels_with_opa_defect/{all_cases[i]}.mha')
        print(f'{labelstr}/la_trans_corrected_{all_cases[i]}_{i:03}.mha')
        GTlabel = sitk.ReadImage(
            f'{source_dir}GTlabels_with_opa_defect/{all_cases[i]}.mha'
        )
        sitk.WriteImage(
            GTlabel,
            f'{labelstr}/la_lables_opa_defect_{all_cases[i]}_{i:03}.mha',
            useCompression=True,
        )

    # Labels :
    # 0: Hors de l'Atrium
    # 1: Corps
    # 2: LAA (Auricule)
    # 3: RSPV (Veines)
    # 4: RIPV (Veines)
    # 5: LIPV (Veines)
    # 6: LSPV (Veines)
    # 7: opacification_defect (calcul à la main = différence entre prédiction D2 et label auricule)

    generate_dataset_json(
        output_folder=join(nnunet_dir, dataset_name),
        channel_names={0: 'CT'},
        labels={
            'background': 0,
            'whole_atrium': [1, 2, 3, 4, 5, 6, 7],
            'corps_and_auricle': [1, 2, 7],
            'auricle': [2, 7],
            'opacification_defect': [7],
        },
        num_training_cases=len(all_cases),
        file_ending='.mha',
        dataset_name=dataset_name,
        regions_class_order=[
            1,
            2,
            3,
            4,
        ],  # ['whole_atrium', 'corps_and_auricle', 'auricle', 'opacification_defect]
    )
