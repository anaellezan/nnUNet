import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm
import metrics

# def compute_dice_coefficient(mask_gt, mask_pred):
#     """Computes soerensen-dice coefficient.

#     compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
#     and the predicted mask `mask_pred`.

#     Args:
#       mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
#       mask_pred: 3-dim Numpy array of type bool. The predicted mask.

#     Returns:
#       the dice coeffcient as float. If both masks are empty, the result is NaN.
#     """
#     volume_sum = mask_gt.sum() + mask_pred.sum()
#     if volume_sum == 0:
#         return np.NaN
#     volume_intersect = (mask_gt & mask_pred).sum()
#     return 2*volume_intersect / volume_sum 


# def compute_IoU_coefficient(mask_gt, mask_pred):
#     """Computes IoU coefficient.

#     compute the IoU coefficient between the ground truth mask `mask_gt`
#     and the predicted mask `mask_pred`.

#     Args:
#       mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
#       mask_pred: 3-dim Numpy array of type bool. The predicted mask.

#     Returns:
#       the IoU coeffcient as float. If both masks are empty, the result is NaN.
#     """
#     volume_union = (mask_gt | mask_pred).sum()
#     if volume_union == 0:
#         return np.NaN
#     volume_intersect = (mask_gt & mask_pred).sum()
#     return volume_intersect / volume_union 



def get_dices_list(pred_files, ref_files):
    dices = []
    for i in tqdm(range(len(pred_files))):
        seg_ref = sitk.ReadImage(ref_files[i])
        seg_pred = sitk.ReadImage(pred_files[i])
        seg_ref_arr = sitk.GetArrayFromImage(seg_ref)
        seg_pred_arr = sitk.GetArrayFromImage(seg_pred)
        dice = metrics.compute_dice_coefficient(seg_ref_arr, seg_pred_arr)
        dices.append(dice)
    return dices


def get_IoU_list(pred_files, ref_files):
    IoUs = []
    for i in tqdm(range(len(pred_files))):
        seg_ref = sitk.ReadImage(ref_files[i])
        seg_pred = sitk.ReadImage(pred_files[i])
        seg_ref_arr = sitk.GetArrayFromImage(seg_ref)
        seg_pred_arr = sitk.GetArrayFromImage(seg_pred)
        IoU = metrics.compute_IoU_coefficient(seg_ref_arr, seg_pred_arr)
        IoUs.append(IoU)
    return IoUs

def main():
    path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset001_LA_CT01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/'
    # path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset002_LA_CT00/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/'
    files = [f for f in os.listdir( path ) if f[-4:]=='.mha']

    #pred_files = ['/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset001_LA_CT01/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/' + f for f in files]
    pred_files_ensemble = ['/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset001_LA_CT01/TEST_ENSEMBLE/' + f for f in files]
    ref_files = ['/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset001_LA_CT01/labelsTr/' + f for f in files]
    # pred_files_ensemble = ['/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset002_LA_CT00/TEST_ENSEMBLE/' + f for f in files]
    # ref_files = ['/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset002_LA_CT00/labelsTr/' + f for f in files]
    
    
    # dices_ensemble = get_dices_list(pred_files_ensemble, ref_files)
    # print(f'All dices of ensemble predictions for CT01:\n\n{dices_ensemble}\n')
    # print(f'Average Dice: {np.mean(dices_ensemble)}\n\n')

    IoUs_ensemble = get_IoU_list(pred_files_ensemble, ref_files)
    print(f'All IoU of ensemble predictions for CT00:\n\n{IoUs_ensemble}\n')
    print(f'Average IoU: {np.mean(IoUs_ensemble)}\n\n')

if __name__ == '__main__':
    main()