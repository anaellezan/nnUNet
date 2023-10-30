import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes





if __name__ == "__main__":
    nnunet_dir = '/media/sharedata/myosaiq_challenge/ana_test/nnUNet_raw/raw_data'

    #dataset_name = 'Dataset001_D8'
    # dataset_name = 'Dataset002_M1'
    dataset_name = 'Dataset003_M12'
    imagestr = join(nnunet_dir, dataset_name, 'imagesTr')
    labelstr = join(nnunet_dir, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)


    train_source = join(nnunet_dir, 'training')

    #source_dir = join(nnunet_dir, 'D8_postMI')
    # source_dir = join(nnunet_dir, 'M1_postMI')
    source_dir = join(nnunet_dir, 'M12_postMI')
    
    source_images_dir = join(source_dir, 'images')
    source_labels_dir = join(source_dir, 'labels')


    # convert 4d train images
    source_images = [i for i in subfiles(source_images_dir, suffix='.nii.gz', join=False) if
                        not i.startswith('.') and not i.startswith('_')]
    
    for s in source_images:
        # copy images
        print(join(imagestr, s[:-7] + '_0000.nii.gz'))
        shutil.copy(join(source_images_dir, s), join(imagestr, s[:-7] + '_0000.nii.gz'))

    # copy labels
    source_labels = [i for i in subfiles(source_labels_dir, suffix='.nii.gz', join=False) if
                        not i.startswith('.') and not i.startswith('_')]
   
    for s in source_labels:
        print(join(labelstr, s))
        shutil.copy(join(source_labels_dir, s), join(labelstr, s))



    generate_dataset_json(output_folder=join(nnunet_dir, dataset_name),
                          #channel_names={0: 'MRI_D8'},
                        #   channel_names={0: 'MRI_M1'},
                          channel_names={0: 'MRI_M12'},
                          labels={'background': 0, 'lv': [1, 2, 3, 4], 'myocardium':[2, 3, 4], 'total_infarct': [3,4], 'mvo':[4] },
                          num_training_cases=len(source_images),
                          file_ending='.nii.gz',
                          dataset_name=dataset_name,
                          regions_class_order=[1, 2, 3, 4], # ['myocardium', 'total_infarct', 'mvo']
                          )
