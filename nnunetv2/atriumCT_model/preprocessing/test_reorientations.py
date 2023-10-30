from matplotlib.pyplot import imshow
import os
import SimpleITK as sitk
import time


folder = '/media/sharedata/atriumCT/tests_on_reorientation/AFCT_17_07_03_190141/'

im0 = sitk.ReadImage('/media/sharedata/atriumCT/data_anonymized/AFCT_17_07_03_190141/00. ctData.mha')

im1 = sitk.ReadImage('/media/sharedata/atriumCT/data_anonymized/AFCT_17_07_03_190141/01. ctData.mha')

mask = sitk.ReadImage('/media/sharedata/atriumCT/data_anonymized/AFCT_17_07_03_190141/07. leftAtriumMask.mha')

start_time = time.time()

mask_resampled_nearestNeighbor = sitk.Resample(mask, im0, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
mask_twice_resampled_nearestNeighbor = sitk.Resample(mask_resampled_nearestNeighbor, im1, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())

# mask_resampled_labelGaussian = sitk.Resample(mask, im0, sitk.Transform(), sitk.sitkLabelGaussian, 0, mask.GetPixelID())

# mask_twice_resampled_labelGaussian = sitk.Resample(mask_resampled_labelGaussian, im1, sitk.Transform(), sitk.sitkLabelGaussian, 0, mask.GetPixelID())

for i in range(99):
    # mask_resampled_labelGaussian = sitk.Resample(mask_twice_resampled_labelGaussian, im0, sitk.Transform(), sitk.sitkLabelGaussian, 0, mask.GetPixelID())
    # mask_twice_resampled_labelGaussian = sitk.Resample(mask_resampled_labelGaussian, im1, sitk.Transform(), sitk.sitkLabelGaussian, 0, mask.GetPixelID())
    mask_resampled_nearestNeighbor = sitk.Resample(mask_twice_resampled_nearestNeighbor, im0, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    mask_twice_resampled_nearestNeighbor = sitk.Resample(mask_resampled_nearestNeighbor, im1, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())

# print("Took %s seconds to reorient 100 times the LabelGaussian mask." % (time.time() - start_time))
print("Took %s seconds to reorient 100 times the nearestNeighbor mask." % (time.time() - start_time))