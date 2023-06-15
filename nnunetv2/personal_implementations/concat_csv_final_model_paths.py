import pandas as pd

# NB_FOLDS = 5
NB_FOLDS = 10

# path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_results/Dataset001_D8/nnUNetTrainer__nnUNetPlans__3d_fullres/'
path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_results/Dataset010_D8/nnUNetTrainer__nnUNetPlans__3d_fullres/'
fold = 0
df = pd.read_csv(f'{path}fold_{fold}/paths.csv')
dfs = []
global_path = f'{path}paths.csv'
for fold in range(1, NB_FOLDS):
    dfs.append(pd.read_csv(f'{path}fold_{fold}/paths.csv'))
    
print(dfs)
pd.concat(dfs).to_csv(global_path, index=False)