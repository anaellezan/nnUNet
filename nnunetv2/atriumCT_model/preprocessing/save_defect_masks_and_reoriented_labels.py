import os
import SimpleITK as sitk
from tqdm import tqdm


def get_intersection(mask_1, mask_2):
    return (mask_1 & mask_2)

def get_union(mask_1, mask_2):
    return (mask_1 | mask_2)


dataset = 'Dataset004_LA_CT00_corrected_voted'
path = f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/'
files = [f for f in os.listdir( path ) if f[-4:]=='.mha']

pred_files = [f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/prediction_3d_dataset002/' + f for f in files]
#ref_files = [f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/{dataset}/labelsTr/' + f for f in files]
ref_label_files = ['/media/sharedata/atriumCT/corrected_data/GTlabels/' + f[-11:-8].split('_')[-1] + '.mha' for f in files]


def main():
    for i in tqdm(range(len(pred_files))):

        print(pred_files[i])
        seg_pred = sitk.ReadImage(pred_files[i])
        seg_ref_label =  sitk.ReadImage(ref_label_files[i])

        origin_id = int(pred_files[i][-11:-8].split('_')[-1])
        nnunet_id = int(pred_files[i][-7:-4])
        print(origin_id)

        seg_pred_arr = sitk.GetArrayFromImage(seg_pred)
        seg_ref_label.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

        seg_ref_label_to_seg_pred = sitk.Resample(seg_ref_label, seg_pred, sitk.Transform(), sitk.sitkLabelGaussian, 0, seg_ref_label.GetPixelID())
        seg_ref_label_to_seg_pred_arr = sitk.GetArrayFromImage(seg_ref_label_to_seg_pred)
        print(seg_ref_label_to_seg_pred_arr.shape)

        # Save reoriented labels
        sitk.WriteImage(seg_ref_label_to_seg_pred, f'/media/sharedata/atriumCT/corrected_data/GTlabels_reoriented/{origin_id}.mha' ,useCompression=True)


        seg_ref_auricule_arr = (seg_ref_label_to_seg_pred_arr == 2).astype(int)

        intersec_pred_auricle = get_intersection(seg_pred_arr, seg_ref_auricule_arr)
        union_pred_auricle = get_union(seg_pred_arr, seg_ref_auricule_arr)

        # the logical operation of (the union between the LA prediction and the auricle label) minus the prediction
        # gives the error, that is to say: the opacification defect
        opacification_defect_arr = (union_pred_auricle - seg_pred_arr)

        opacification_defect_mask = sitk.GetImageFromArray(opacification_defect_arr)
        opacification_defect_mask = sitk.Cast(opacification_defect_mask, sitk.sitkUInt8) # Cast necessary to be opened in MUSIC
        opacification_defect_mask.CopyInformation(seg_pred)

        # Save opacification defect masks
        sitk.WriteImage(opacification_defect_mask,f'/media/sharedata/atriumCT/corrected_data/GTopacification_defect/{origin_id}.mha' ,useCompression=True)



if __name__ == "__main__":
    main()