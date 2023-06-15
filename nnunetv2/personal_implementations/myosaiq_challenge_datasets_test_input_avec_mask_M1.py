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
import numpy as np


def convert_labels(label_path):
    label = sitk.ReadImage(label_path)
    label_npy = sitk.GetArrayFromImage(label)

    uniques = np.unique(label_npy)
    for u in uniques:
        if u not in [0, 1, 2, 3]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(label_npy)
    seg_new[label_npy == 1] = 0
    seg_new[label_npy == 2] = 0
    seg_new[label_npy == 3] = 1

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(label)
    return img_corr

FOLDS_NUMBER = 10

if __name__ == "__main__":
    # nnunet_dir = '/media/sharedata/myosaiq_challenge/ana_test/nnUNet_raw/raw_data'
    nnunet_dir = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training'
    # dataset_name = 'Dataset007_D8_5FOLDS'
    # dataset_name = 'Dataset008_M1_5FOLDS'
    # dataset_name = 'Dataset009_M12_FOLDS'
    # dataset_name = 'Dataset010_D8'
    # dataset_name = 'Dataset011_M1'
    # dataset_name = 'Dataset012_M12'
    # dataset_name = 'Dataset013_D8'
    # dataset_name = 'Dataset014_M1'
    # dataset_name = 'Dataset015_M12'
    dataset_name = 'Dataset017_M1_mask'
    # dataset_name = 'Dataset018_M12_mask'
    # dataset_name = 'Dataset004_D8CT'
    # dataset_name = 'Dataset005_M1CT'
    # dataset_name = 'Dataset006_M12CT'
    imagestr = join(nnunet_dir, dataset_name, 'imagesTr')
    labelstr = join(nnunet_dir, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # dataset_masks = 'Dataset010_D8'
    dataset_masks = 'Dataset011_M1'
    # dataset_masks = 'Dataset012_M12'

    train_source = join(nnunet_dir, 'training')

    # source_dir = join(nnunet_dir, 'D8_postMI')
    source_dir = join(nnunet_dir, 'M1_postMI')
    # source_dir = join(nnunet_dir, 'M12_postMI')
    
    source_images_dir = join(source_dir, 'images')
    source_labels_dir = join(source_dir, 'labels')


    source_images = [i for i in subfiles(source_images_dir, suffix='.nii.gz', join=False) if
                        not i.startswith('.') and not i.startswith('_')]
    
    for s in source_images:
        # copy images
        print(join(imagestr, s[:-7] + '_0000.nii.gz'))
        shutil.copy(join(source_images_dir, s), join(imagestr, s[:-7] + '_0000.nii.gz'))


    for fold in range(FOLDS_NUMBER):
      source_pred_dir = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_results/{dataset_masks}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{fold}/validation/'
      source_predictions = [i for i in subfiles(source_pred_dir, suffix='.nii.gz', join=False) if
                        not i.startswith('.') and not i.startswith('_')]
    
      for s in source_predictions:
          # copy images
          print(join(imagestr, s[:-7] + '_0001.nii.gz'))
          shutil.copy(join(source_pred_dir, s), join(imagestr, s[:-7] + '_0001.nii.gz'))


    # copy labels
    source_labels = [i for i in subfiles(source_labels_dir, suffix='.nii.gz', join=False) if
                        not i.startswith('.') and not i.startswith('_')]
   

    for s in source_labels:
        print(join(labelstr, s))
        label_path = join(source_labels_dir, s)
        new_labels = convert_labels(label_path)
        sitk.WriteImage(new_labels, join(labelstr, s))


    generate_dataset_json(output_folder=join(nnunet_dir, dataset_name),
                        #   channel_names={0: 'MRI_D8'},
                          # channel_names={0: 'MRI_M1'},
                          channel_names={0: 'MRI',
                          1: 'seg'},
                        #   channel_names={0: 'CT'},
                           labels={'background': 0,  'total_infarct': [1]},
                          num_training_cases=len(source_images),
                          file_ending='.nii.gz',
                          dataset_name=dataset_name,
                          regions_class_order=[1], # ['myocardium', 'total_infarct', 'mvo']
                          )

