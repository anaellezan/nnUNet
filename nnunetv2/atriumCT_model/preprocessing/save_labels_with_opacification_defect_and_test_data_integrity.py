""" This script:
- Makes sure that images and masks have same origin, spacing, etc.,
otherwise, it resamples it into opacification mask file parameters (seg_opa_file)
- Fills the big holes in the body label
- Filters on auricle label so that we can fill the holes and get a smooth delimitation
between the body and the auricle
- Adds opacification_defect label
- Checks that images, masks and labels are not empty and don't contain a unique value
"""

import gc
import os

from nnunetv2.atriumCT_model.preprocessing import utils
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def main():

    path = '/media/sharedata/atriumCT/corrected_data/GTlabels_reoriented/'
    ref_label_files = [path + f for f in os.listdir(path)]

    for i in tqdm(range(33, len(ref_label_files))):

        print(ref_label_files[i])
        seg_ref_label = sitk.ReadImage(ref_label_files[i])
        origin_id = int(ref_label_files[i].split('/')[-1][:-4])
        mask_file = (
            f'/media/sharedata/atriumCT/corrected_data/GTmasks2/{origin_id}.mha'
        )
        print(mask_file)
        seg_mask = sitk.ReadImage(mask_file)
        seg_opa_file = f'/media/sharedata/atriumCT/corrected_data/GTopacification_defect/{origin_id}.mha'
        print(seg_opa_file)
        seg_opa = sitk.ReadImage(seg_opa_file)

        if seg_ref_label.GetSize() != seg_opa.GetSize():
            seg_ref_label = sitk.Resample(
                seg_ref_label,
                seg_opa,
                sitk.Transform(),
                sitk.sitkLabelGaussian,
                0,
                seg_ref_label.GetPixelID(),
            )

        if seg_mask.GetSize() != seg_opa.GetSize():
            seg_mask = sitk.Resample(
                seg_mask,
                seg_opa,
                sitk.Transform(),
                sitk.sitkLabelGaussian,
                0,
                seg_mask.GetPixelID(),
            )

        seg_mask_arr = sitk.GetArrayFromImage(seg_mask)
        seg_opa_arr = sitk.GetArrayFromImage(seg_opa)
        seg_ref_label_arr = sitk.GetArrayFromImage(seg_ref_label)

        # Put labels values on the total mask. It enables to fill the big holes in the body
        filled_seg_ref_label_arr = np.where(
            seg_ref_label_arr != 0, seg_ref_label_arr, seg_mask_arr
        )

        # Filter on auricle label so that we can fill the holes and get a smooth delimitation
        # between the body and the auricle
        auricle = seg_ref_label == 2

        closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        closing_filter.SetKernelType(sitk.sitkBall)
        closing_filter.SetKernelRadius(int(min(auricle.GetSize()) / 4))
        auricle_closed = closing_filter.Execute(auricle)
        auricle_closed_arr = sitk.GetArrayFromImage(auricle_closed)
        auricle_closed_arr = auricle_closed_arr * seg_mask_arr
        seg_ref_label_closed_arr = np.where(
            auricle_closed_arr != 0, 2, filled_seg_ref_label_arr
        )

        # Adding opacification_defect label
        filled_seg_ref_label_with_opa_defect_arr = np.where(
            seg_opa_arr != 0, seg_opa_arr * 7, seg_ref_label_closed_arr
        ).astype(int)

        filled_seg_ref_label_with_opa_defect_mask = sitk.GetImageFromArray(
            filled_seg_ref_label_with_opa_defect_arr
        )
        filled_seg_ref_label_with_opa_defect_mask = sitk.Cast(
            filled_seg_ref_label_with_opa_defect_mask, sitk.sitkUInt8
        )  # Cast necessary to be opened in MUSIC
        filled_seg_ref_label_with_opa_defect_mask.CopyInformation(seg_mask)

        filled_seg_ref_label_with_opa_defect_arr_path = f'/media/sharedata/atriumCT/corrected_data/GTlabels_with_opa_defect/{origin_id}.mha'
        utils.test_empty_arr(
            filled_seg_ref_label_with_opa_defect_arr,
            filled_seg_ref_label_with_opa_defect_arr_path,
            2,
        )
        utils.test_unique_val(
            filled_seg_ref_label_with_opa_defect_arr,
            filled_seg_ref_label_with_opa_defect_arr_path,
            8,
        )

        # Save label masks containing opacification_defect label equal to 7
        sitk.WriteImage(
            filled_seg_ref_label_with_opa_defect_mask,
            filled_seg_ref_label_with_opa_defect_arr_path,
            useCompression=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
