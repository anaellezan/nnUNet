
import pandas as pd
from os import listdir, remove
from os.path import isfile, join, exists

# NB_FOLDS = 5
# NB_FOLDS = 10
NB_FOLDS = 8

# dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset001_D8/'
# dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset006_M12CT/'
# dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset014_M1/'
# dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset004_D8CT/'
# dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/Dataset017_M1_mask/'
dataset_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/training/'#Dataset017_M1_mask/'


if __name__ == "__main__":
  # global_path = f'{dataset_path}val_data/paths.csv'
  # global_path = f'{dataset_path}val_data/paths_1000.csv'
  global_path = f'{dataset_path}/Dataset017_M1_mask/val_data/paths_1000.csv'
  if exists(global_path):
    remove(global_path)

  for fold in range(NB_FOLDS):
    # prediction_path = f'{dataset_path}val_data/fold_{fold}/prediction/'
    # prediction_path = f'{dataset_path}val_data/fold_{fold}/prediction_1000_best/'
    prediction_path = f'{dataset_path}/Dataset017_M1_mask/val_data/fold_{fold}/prediction_output/'

    fold_ref_path = f'{dataset_path}/Dataset011_M1/val_data/fold_{fold}/labelsTr/'
  # fold_path = training_path + f'fold_{fold}/'
  # fold_val_path = fold_path + 'validation/'

    gz_files_targets = [f for f in listdir(prediction_path) if (isfile(join(prediction_path, f)) & (f[-3:]=='.gz'))]
    gz_folder_targets = [prediction_path + f for f in gz_files_targets]
    print(gz_folder_targets)
    print()

    gz_folder_refs = [fold_ref_path + f for f in gz_files_targets]
    print(gz_folder_refs)
    print()
    df = pd.DataFrame({'REFERENCE': gz_folder_refs,
    'TARGET': gz_folder_targets})
    print(df)
    # df.to_csv(f'{dataset_path}val_data/fold_{fold}/paths.csv', index=False)

    df.to_csv(f'{dataset_path}/Dataset017_M1_mask/val_data/fold_{fold}/paths_1000.csv', index=False)

    # global_path = f'{dataset_path}val_data/paths.csv'
    if not exists(global_path):
      df.to_csv(global_path, index=False)
    else:
      df2 = pd.read_csv(global_path)
      print(pd.concat([df2, df]))
      pd.concat([df2, df]).to_csv(global_path, index=False)


  
  

