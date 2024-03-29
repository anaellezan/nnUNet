import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import metrics

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


# dataset = 'Dataset004_LA_CT00_corrected_voted'
# # path = f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/'
# # path = f'/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset004_LA_CT00_corrected_voted/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres/fold_0/validation/'
# # path = f'/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset004_LA_CT00_corrected_voted/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_lr_06_1500_epochs/fold_0/validation/'
# # path = f'/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset004_LA_CT00_corrected_voted/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_1er_test/fold_0/validation/'
# # path = f'/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset005_LA_CT00_corrected_voted_region/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres/fold_0/prediction_500/'
# path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset005_LA_CT00_corrected_voted_region/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_500_epochs/fold_0/validation_best_500/'
# # path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset006_LA_CT00_region_fenetrage/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres/fold_0/training_1000_epochs/validation/'
# # path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset006_LA_CT00_region_fenetrage/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_classic_loss/fold_0/training_1000_epochs/validation_872_best_1000/'
# # path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset006_LA_CT00_region_fenetrage/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_classic_loss/fold_0/validation/'
# # path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset006_LA_CT00_region_fenetrage/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_classic_loss/fold_0/validation/'
# files = [f for f in os.listdir( path ) if f[-4:]=='.mha']

# #pred_files = [f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/' + f for f in files]
# # pred_files = [f'/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset004_LA_CT00_corrected_voted/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres/fold_0/validation/' + f for f in files if 'mha' in f]
# pred_files = [path + f for f in files if 'mha' in f]

# print(len(pred_files))


path = '/media/sharedata/atriumCT/atrium_medFormer/results/LA_CT00/fold_0_val/mha_files/'

files = [f for f in os.listdir(path) if f[-4:] == '.mha']

pred_files = [path + f for f in files if '.mha' in f]

print(len(pred_files))


# la_trans_corrected_6_000
# la_lables_opa_defect_106_065

# ref_files = [f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/labelsTr_old_not_resampled/' + f for f in files]
ref_files = [
    f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset004_LA_CT00_corrected_voted/labelsTr/'
    + f.replace('la_lables_opa_defect', 'la_trans_corrected').replace(
        '_0.mha', '.mha'
    )
    for f in files
]
print(len(ref_files))


# ref_label_files = ['/media/sharedata/atriumCT/corrected_data/GTlabels/' + f[-11:-8].split('_')[-1] + '.mha' for f in files]
ref_label_files = [
    '/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset005_LA_CT00_corrected_voted_region/labelsTr/'
    + f.replace('_0.mha', '.mha')
    for f in files
]
print(len(ref_label_files))


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def main():

    # df = pd.read_csv(f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/patients_history.csv')
    # df = pd.read_csv(f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset004_LA_CT00_corrected_voted/prediction_3d_dataset002/patients_history.csv')

    ids = [int(file.split('_')[-3]) for file in files]
    ids_medformer = [int(file.split('_')[-2]) for file in files]
    df = pd.DataFrame({'ID': ids, 'medformer_id': ids_medformer})
    print(df)

    df['medformer_id'] = np.NaN
    df['Dice'] = np.NaN
    df['IoU'] = np.NaN
    df['Contour_Volume'] = np.NaN

    df['New_dice_auricle'] = np.NaN
    df['Dice_auricle'] = np.NaN
    df['Dice_corps'] = np.NaN
    df['Dice_RSPV_1'] = np.NaN
    df['Dice_RIPV_2'] = np.NaN
    df['Dice_LIPV_3'] = np.NaN
    df['Dice_LSPV_4'] = np.NaN
    df['Contour_vol_auricle'] = np.NaN

    df['Spacing_ref_0'] = np.NaN
    df['Spacing_ref_1'] = np.NaN
    df['Spacing_ref_2'] = np.NaN
    df['vol_auricle_ref'] = np.NaN
    df['vol_opa_ref'] = np.NaN
    df['vol_opa_pred'] = np.NaN

    for i in tqdm(range(len(pred_files))):
        print(ref_files[i])
        seg_ref = sitk.ReadImage(ref_files[i])
        print(pred_files[i])
        seg_pred = sitk.ReadImage(pred_files[i])
        seg_ref_label = sitk.ReadImage(ref_label_files[i])
        print(seg_ref.GetSize())
        print(seg_pred.GetSize())
        print(seg_ref_label.GetSize())

        origin_id = int(pred_files[i].split('_')[-3])
        medformer_id = int(pred_files[i].split('_')[-2])
        print(origin_id)
        seg_pred_resampled_labelGaussian = sitk.Resample(
            seg_pred,
            seg_ref,
            sitk.Transform(),
            sitk.sitkLabelGaussian,
            0,
            seg_pred.GetPixelID(),
        )

        print(seg_pred_resampled_labelGaussian.GetSize())
        seg_ref_arr = sitk.GetArrayFromImage(seg_ref).astype(int)
        seg_pred_arr = sitk.GetArrayFromImage(seg_pred_resampled_labelGaussian)
        seg_ref_label_arr = sitk.GetArrayFromImage(seg_ref_label)
        seg_ref_auricule_arr = (seg_ref_label_arr == 2) | (
            seg_ref_label_arr == 7
        )
        seg_ref_opa_defect_arr = seg_ref_label_arr == 7
        seg_pred_opa_defect_arr = seg_pred_arr == 4

        # /!\
        # If seg pred is a labeled prediction:
        seg_pred_arr = seg_pred_arr != 0  # Otherwise comment

        seg_ref_corps_arr = seg_ref_label_arr == 1
        seg_ref_rspv_1_arr = seg_ref_label_arr == 3
        seg_ref_ripv_2_arr = seg_ref_label_arr == 4
        seg_ref_lipv_3_arr = seg_ref_label_arr == 5
        seg_ref_lspv_4_arr = seg_ref_label_arr == 6

        contour_vol = metrics.compute_contour_volume(seg_ref_arr, seg_pred_arr)
        dice = metrics.compute_dice_coefficient(seg_ref_arr, seg_pred_arr)
        IoU = metrics.compute_IoU_coefficient(seg_ref_arr, seg_pred_arr)

        # Compute metrics on every label

        pred_auricule_intersection = metrics.get_intersection(
            seg_pred_arr, seg_ref_auricule_arr
        )
        pred_corps_intersection = metrics.get_intersection(
            seg_pred_arr, seg_ref_corps_arr
        )
        pred_rspv_1_intersection = metrics.get_intersection(
            seg_pred_arr, seg_ref_rspv_1_arr
        )
        pred_ripv_2_intersection = metrics.get_intersection(
            seg_pred_arr, seg_ref_ripv_2_arr
        )
        pred_lipv_3_intersection = metrics.get_intersection(
            seg_pred_arr, seg_ref_lipv_3_arr
        )
        pred_lspv_4_intersection = metrics.get_intersection(
            seg_pred_arr, seg_ref_lspv_4_arr
        )

        if len(np.unique(seg_ref_auricule_arr)) > 1:
            bbox_auricle_coord = bbox2_3D(seg_ref_auricule_arr)
            bbox_auricle = np.zeros(np.shape(seg_ref_auricule_arr))
            bbox_auricle[
                bbox_auricle_coord[0] : bbox_auricle_coord[1] + 1,
                bbox_auricle_coord[2] : bbox_auricle_coord[3] + 1,
                bbox_auricle_coord[4] : bbox_auricle_coord[5] + 1,
            ] = 1
            multiplication_bbox_auricle_seg_pred = (
                bbox_auricle * seg_pred_arr
            ).astype(int)
            multiplication_bbox_auricle_seg_ref = (
                bbox_auricle * seg_ref_arr
            ).astype(int)
            new_dice_auricle = metrics.compute_dice_coefficient(
                multiplication_bbox_auricle_seg_pred,
                multiplication_bbox_auricle_seg_ref,
            )
            print(new_dice_auricle)
            df.loc[df['ID'] == origin_id, 'New_dice_auricle'] = new_dice_auricle
            contour_volume_auricle = metrics.compute_contour_volume(
                multiplication_bbox_auricle_seg_pred,
                multiplication_bbox_auricle_seg_ref,
            )

        dice_auricle = metrics.compute_dice_coefficient(
            seg_ref_auricule_arr, pred_auricule_intersection
        )
        dice_corps = metrics.compute_dice_coefficient(
            seg_ref_corps_arr, pred_corps_intersection
        )
        dice_rspv_1 = metrics.compute_dice_coefficient(
            seg_ref_rspv_1_arr, pred_rspv_1_intersection
        )
        dice_ripv_2 = metrics.compute_dice_coefficient(
            seg_ref_ripv_2_arr, pred_ripv_2_intersection
        )
        dice_lipv_3 = metrics.compute_dice_coefficient(
            seg_ref_lipv_3_arr, pred_lipv_3_intersection
        )
        dice_lspv_4 = metrics.compute_dice_coefficient(
            seg_ref_lspv_4_arr, pred_lspv_4_intersection
        )

        df.loc[df['ID'] == origin_id, 'medformer_id'] = medformer_id
        df.loc[df['ID'] == origin_id, 'Contour_Volume'] = contour_vol
        df.loc[df['ID'] == origin_id, 'Dice'] = dice
        df.loc[df['ID'] == origin_id, 'IoU'] = IoU

        df.loc[df['ID'] == origin_id, 'Dice_auricle'] = dice_auricle
        df.loc[df['ID'] == origin_id, 'Dice_corps'] = dice_corps
        df.loc[df['ID'] == origin_id, 'Dice_RSPV_1'] = dice_rspv_1
        df.loc[df['ID'] == origin_id, 'Dice_RIPV_2'] = dice_ripv_2
        df.loc[df['ID'] == origin_id, 'Dice_LIPV_3'] = dice_lipv_3
        df.loc[df['ID'] == origin_id, 'Dice_LSPV_4'] = dice_lspv_4
        df.loc[df['ID'] == origin_id, 'Contour_vol_auricle'] = (
            contour_volume_auricle
        )

        df.loc[df['ID'] == origin_id, 'Spacing_ref_0'] = seg_ref.GetSpacing()[0]
        df.loc[df['ID'] == origin_id, 'Spacing_ref_1'] = seg_ref.GetSpacing()[1]
        df.loc[df['ID'] == origin_id, 'Spacing_ref_2'] = seg_ref.GetSpacing()[2]

        df.loc[df['ID'] == origin_id, 'vol_auricle_ref'] = (
            seg_ref_auricule_arr.sum()
        )
        df.loc[df['ID'] == origin_id, 'vol_opa_ref'] = (
            seg_ref_opa_defect_arr.sum()
        )
        df.loc[df['ID'] == origin_id, 'vol_opa_pred'] = (
            seg_pred_opa_defect_arr.sum()
        )
        print(df)

    print(df)

    df.to_csv(f'{path}patients_history_metrics_etude_corr.csv', index=False)

    # df.to_csv(f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/patients_history_final_metrics_new_dice_corrected.csv', index=False)
    df['medformer_id'] = df['medformer_id'].astype('int')
    df.to_csv(f'{path}patients_history_metrics_etude_corr.csv', index=False)
    # df.to_csv(f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/patients_history_final_metrics_new_dice_corrected.csv', index=False)


if __name__ == '__main__':
    main()
