import SimpleITK as sitk
import shutil
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import *

FOLDS_NUMBER = 10

masks_m11 = []
# dataset = 'Dataset017_M1_mask'
dataset = 'Dataset018_M12_mask'
masks_m11_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset}/imagesTs/'
preds_m17_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset}/pred_test_best/'
preds_output_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset}/prediction_output/'

maybe_mkdir_p(preds_output_path)

# convert 4d train images
masks_m11_fold = [i for i in subfiles(masks_m11_path, suffix='.nii.gz', join=False) if
                    not i.startswith('.') and not i.startswith('_') and '0001' in i]

preds_m17_fold = [i for i in subfiles(preds_m17_path, suffix='.nii.gz', join=False) if
                    not i.startswith('.') and not i.startswith('_')]

for i in range(len(masks_m11_fold)):
    print(masks_m11_fold[i])
    mask_m11_path = masks_m11_path + masks_m11_fold[i]
    mask_m11 = sitk.ReadImage(mask_m11_path)
    mask_m11_npy = sitk.GetArrayFromImage(mask_m11)
    print(np.unique(mask_m11_npy))
    print(mask_m11.GetSize())


    print(preds_m17_fold[i])

    pred_m17_path = preds_m17_path + preds_m17_fold[i]
    pred_m17 = sitk.ReadImage(pred_m17_path)
    pred_m17_npy = sitk.GetArrayFromImage(pred_m17)
    print(np.unique(pred_m17_npy))
    print(pred_m17.GetSize())

    seg_new = np.zeros_like(mask_m11_npy)

    seg_new[mask_m11_npy == 1] = 1
    seg_new[mask_m11_npy == 2] = 2
    seg_new[mask_m11_npy == 3] = 2
    seg_new[pred_m17_npy == 1] = 3

    # seg_new.shape
    img_new = sitk.GetImageFromArray(seg_new)
    img_new.CopyInformation(mask_m11)
    sitk.WriteImage(img_new, preds_output_path + preds_m17_fold[i])