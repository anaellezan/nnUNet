
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.utilities.file_and_folder_operations import load_json
import shutil
import os


# /!\ ATTENTION QUE 7 folds
NB_FOLDS = 7
# NB_FOLDS = 10
# NB_FOLDS = 5

# dataset = 'Dataset004_D8CT'
# dataset = 'Dataset017_M1_mask'
# dataset_input = 'Dataset011_M1'
dataset = 'Dataset016_D8CT_mask'
dataset_input = 'Dataset010_D8'


# splits_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_preprocessed/Dataset001_D8/splits_final.json'
# splits_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_preprocessed/Dataset012_M12/splits_final.json'
# splits_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_preprocessed/Dataset006_M12CT/splits_final.json'
# splits_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_preprocessed/Dataset014_M1/splits_final.json'
# splits_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_preprocessed/Dataset015_M12/splits_final.json'
splits_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_preprocessed/{dataset}/splits_final.json'
splits = load_json(splits_path)
# raw_dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset001_D8/'
# raw_dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset012_M12/'
raw_dataset_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset}/'
raw_imagesTr_path = raw_dataset_path + 'imagesTr/'
raw_imagesTr_01_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset_input}/val_data/'
# raw_imagesTr_01_path = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset_input}/pred_test_best/'
raw_labelsTr_path = raw_dataset_path + 'labelsTr/'


for fold in range(NB_FOLDS):
    directory = f'{raw_dataset_path}val_data/fold_{fold}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    label_dir = directory + 'labelsTr'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    images_dir = directory + 'imagesTr'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    pred_dir = directory + 'prediction'
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
  
    print(splits[fold]['val'])

    print("Using splits from existing split file:", splits_path)
    print("The split file contains %d splits." % len(splits))
    tr_keys = splits[fold]['train']
    val_keys = splits[fold]['val']
    print("This split has %d training and %d validation cases."
                            % (len(tr_keys), len(val_keys)))

    val_imagesTr_paths = [(raw_imagesTr_path + f + '_0000.nii.gz') for f in val_keys]

    val_labelsTr_paths = [(raw_labelsTr_path + f + '.nii.gz') for f in val_keys]
    val_imagesTr_paths_dir = [f.replace('imagesTr/', f'val_data/fold_{fold}/imagesTr/') for f in val_imagesTr_paths]
    val_labelsTr_paths_dir = [f.replace('labelsTr/', f'val_data/fold_{fold}/labelsTr/') for f in val_labelsTr_paths]
    print()
    print(val_labelsTr_paths_dir)
    print()

    for i in range(len(val_imagesTr_paths)):
        shutil.copyfile(val_imagesTr_paths[i], val_imagesTr_paths_dir[i])
        shutil.copyfile(f'{raw_imagesTr_01_path}fold_{fold}/prediction_600_best/{val_keys[i]}.nii.gz', val_imagesTr_paths_dir[i].replace('0000', '0001'))
        print(f'{raw_imagesTr_01_path}fold_{fold}/prediction_1000_best/{val_keys[i]}.nii.gz')
        print(val_imagesTr_paths_dir[i].replace('0000', '0001'))
        print(val_imagesTr_paths_dir[i])
        print()
        shutil.copyfile(val_labelsTr_paths[i], val_labelsTr_paths_dir[i])










# FOLDS_NUMBER = 10

# dataset_input_pred = 'Dataset011_M1'
# dataset_output = 'Dataset017_M1_mask'
# # dataset_input_pred = 'Dataset012_M12'
# # dataset_output = 'Dataset018_M12_mask'

# input_pred_dir = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset_input_pred}/prediction_600_best'
# # output_dir = f'/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/{dataset_output}/imagesTs'


# for fold in range(FOLDS_NUMBER):

#     val_files = [i for i in subfiles(input_pred_dir, suffix='.nii.gz', join=False) if
#                         not i.startswith('.') and not i.startswith('_')]

# input_pred_images = [i for i in subfiles(input_pred_dir, suffix='.nii.gz', join=False) if
#                         not i.startswith('.') and not i.startswith('_')]




# for s in input_pred_images:
#           # copy images
#           print(join(input_pred_dir, s))
#           print(join(output_dir, s[:-7] + '_0001.nii.gz'))
#           print()
#           shutil.copy(join(input_pred_dir, s), join(output_dir, s[:-7] + '_0001.nii.gz'))