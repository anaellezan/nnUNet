
import shutil
from batchgenerators.utilities.file_and_folder_operations import *

# dataset_input_pred = 'Dataset011_M1'
# dataset_output = 'Dataset017_M1_mask'
# dataset_input_pred = 'Dataset012_M12'
# dataset_output = 'Dataset018_M12_mask'
dataset_input_pred = 'Dataset010_D8'
dataset_output = 'Dataset016_D8CT_mask'

input_pred_dir = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset_input_pred}/pred_test_best'
output_dir = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset_output}/imagesTs'


input_pred_images = [i for i in subfiles(input_pred_dir, suffix='.nii.gz', join=False) if
                        not i.startswith('.') and not i.startswith('_')]

for s in input_pred_images:
          # copy images
          print(join(input_pred_dir, s))
          print(join(output_dir, s[:-7] + '_0001.nii.gz'))
          print()
          shutil.copy(join(input_pred_dir, s), join(output_dir, s[:-7] + '_0001.nii.gz'))