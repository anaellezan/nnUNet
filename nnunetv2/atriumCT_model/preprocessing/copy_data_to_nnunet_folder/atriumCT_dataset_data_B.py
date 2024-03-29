''' Building Dataset001_LA_CT01 using CT01 images and LA masks.
'''

import os
import shutil

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
    source_dir = '/media/sharedata/atriumCT/data/'
    nnunet_dir = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/'
    dataset_name = 'Dataset001_LA_CT01'
    imagestr = join(nnunet_dir, dataset_name, 'imagesTr')
    labelstr = join(nnunet_dir, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    all_cases = os.listdir(source_dir)
    print(f'\n{len(all_cases)} cases in the source dir.\n')

    # Copy images
    source_images = [
        source_dir + f + '/01. ctData.mha' for f in all_cases
    ]  # attention cas où il y a des fichiers en double (1)

    # Copy labels
    source_labels = [
        source_dir + f + '/07. leftAtriumMask.mha' for f in all_cases
    ]  # attention cas où il y a des fichiers en double (1)

    for i in tqdm(range(len(source_images))):
        # copy images
        print(source_images[i])
        print(f'{imagestr}/la_{i:03}_0000.mha')
        shutil.copy(source_images[i], f'{imagestr}/la_{i:03}_0000.mha')

    for i in range(len(source_labels)):
        # copy images
        print(source_labels[i])
        print(f'{labelstr}/la_{i:03}.mha')
        shutil.copy(source_labels[i], f'{labelstr}/la_{i:03}.mha')

    generate_dataset_json(
        output_folder=join(nnunet_dir, dataset_name),
        channel_names={0: 'CT'},
        labels={'background': 0, 'LA': 1},
        num_training_cases=len(source_images),
        file_ending='.mha',
        dataset_name=dataset_name,
    )
