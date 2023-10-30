import gc

from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess,
)

import torch.cuda

from nnunetv2.run.run_training import run_training

# Run with CUDA_VISIBLE_DEVICES=X to specify GPU
# CUDA_VISIBLE_DEVICES=1 python run_training_trans.py

# /!\
# DATASET_IDS = [1, 2, 3]
CONFIGURATIONS = ['3d_fullres', '2d', '3d_lowres']
# CONFIGURATIONS = ['3d_fullres','2d']

# /!\
FOLDS_NUMBER = 5
DATASET_ID_TO_TRAIN = 4
CONFIGURATION_TO_TRAIN = ['3d_fullres']#,'2d']
EXPORT_VALIDATION_PROBABILITY = True



def plan_and_preprocess():
    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(dataset_ids=[DATASET_ID_TO_TRAIN], check_dataset_integrity=True)

    # experiment planning
    print('Experiment planning...')
    plan_experiments(dataset_ids=[DATASET_ID_TO_TRAIN],)
    # overwrite_plans_name: Optional[str] = None)

    # preprocessing
    print('Preprocessing...')
    preprocess(
        dataset_ids=[DATASET_ID_TO_TRAIN], configurations=CONFIGURATIONS,
    )


def main():
    plan_and_preprocess()



if __name__ == '__main__':
    main()
