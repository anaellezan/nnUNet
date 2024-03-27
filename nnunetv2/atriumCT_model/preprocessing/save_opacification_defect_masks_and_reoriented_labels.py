import os
import SimpleITK as sitk
from tqdm import tqdm
import utils


def main():

    dataset = 'Dataset004_LA_CT00_corrected_voted'
    path = f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/'
    files = [f for f in os.listdir( path ) if f[-4:]=='.mha']
    pred_files = [f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/' + f for f in files]

    for i in tqdm(range(len(pred_files))): #range(114, 117)):

        print(pred_files[i])
        origin_id = int(pred_files[i][-11:-8].split('_')[-1])
        ref_label_file = f'/media/sharedata/atriumCT/corrected_data/GTlabels/{origin_id}.mha'
        print(ref_label_file)
        mask_file = f'/media/sharedata/atriumCT/corrected_data/GTmasks2/{origin_id}.mha'
        print(mask_file)
        print()


        seg_pred = sitk.ReadImage(pred_files[i])
        seg_ref_label =  sitk.ReadImage(ref_label_file)
        mask = sitk.ReadImage(mask_file)

        seg_pred_arr = sitk.GetArrayFromImage(seg_pred)
        
        seg_ref_label_dir = seg_ref_label.GetDirection()
        mask_dir = mask.GetDirection()
        if seg_ref_label_dir != mask_dir:
            print(f'Change direction of ID: {origin_id}\nFrom {seg_ref_label_dir} to {mask_dir}')
            seg_ref_label.SetDirection(mask.GetDirection())

        seg_ref_label_to_seg_pred = sitk.Resample(seg_ref_label, seg_pred, sitk.Transform(), sitk.sitkLabelGaussian, 0, seg_ref_label.GetPixelID())
        seg_ref_label_to_seg_pred_arr = sitk.GetArrayFromImage(seg_ref_label_to_seg_pred)
        
        label_reoriented_path = f'/media/sharedata/atriumCT/corrected_data/GTlabels_reoriented/{origin_id}.mha'
        utils.test_empty_arr(seg_ref_label_to_seg_pred_arr, label_reoriented_path, 1)
        utils.test_unique_val(seg_ref_label_to_seg_pred_arr, label_reoriented_path, 7)
        # Save reoriented labels
        sitk.WriteImage(seg_ref_label_to_seg_pred, label_reoriented_path, useCompression=True)

        seg_ref_auricule_arr = (seg_ref_label_to_seg_pred_arr == 2).astype(int)

        union_pred_auricle = utils.get_union(seg_pred_arr, seg_ref_auricule_arr)

        # the logical operation of (the union between the LA prediction and the auricle label) minus the prediction
        # gives the error, that is to say: the opacification defect
        opacification_defect_arr = (union_pred_auricle - seg_pred_arr)

        opacification_defect_mask = sitk.GetImageFromArray(opacification_defect_arr)
        opacification_defect_mask = sitk.Cast(opacification_defect_mask, sitk.sitkUInt8) # Cast necessary to be opened in MUSIC
        opacification_defect_mask.CopyInformation(seg_pred)

        opacification_defect_mask_path = f'/media/sharedata/atriumCT/corrected_data/GTopacification_defect/{origin_id}.mha'
        utils.test_empty_arr(opacification_defect_arr, opacification_defect_mask_path, 0.003)
        # Save opacification defect masks
        sitk.WriteImage(opacification_defect_mask, opacification_defect_mask_path, useCompression=True)



if __name__ == "__main__":
    main()