''' LAST VERSION USED FOR PRETRAINING
Training of Dataset002_LA_CT00 with CT00 (transaxial)
'''

import torch.cuda
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess,
)
from nnunetv2.run.run_training import run_training

# Run with CUDA_VISIBLE_DEVICES=X to specify GPU
# CUDA_VISIBLE_DEVICES=1 python run_training_trans.py

# /!\
# DATASET_IDS = [1, 2, 3]
CONFIGURATIONS = ['3d_fullres', '2d', '3d_lowres']
# CONFIGURATIONS = ['3d_fullres','2d']

# /!\
FOLDS_NUMBER = 5
DATASET_ID_TO_TRAIN = 2
CONFIGURATION_TO_TRAIN = ['3d_fullres']  # ,'2d']
EXPORT_VALIDATION_PROBABILITY = True


def plan_and_preprocess():
    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(
        dataset_ids=[DATASET_ID_TO_TRAIN], check_dataset_integrity=True
    )

    # experiment planning
    print('Experiment planning...')
    plan_experiments(
        dataset_ids=[DATASET_ID_TO_TRAIN],
    )
    # overwrite_plans_name: Optional[str] = None)

    # preprocessing
    print('Preprocessing...')
    preprocess(
        dataset_ids=[DATASET_ID_TO_TRAIN],
        configurations=CONFIGURATIONS,
    )


def train(dataset_id, fold, device, conf):

    run_training(
        dataset_name_or_id=dataset_id,
        configuration=conf,
        nb_folds=FOLDS_NUMBER,
        fold=fold,
        export_validation_probabilities=EXPORT_VALIDATION_PROBABILITY,
        device=device,
        num_epochs=500,
        # num_epochs=1000,
        continue_training=True,
    )


def main():
    # plan_and_preprocess()
    device = torch.device('cuda')

    configuration = '3d_fullres'
    print(f"configuration: \n{configuration}\n")
    print(f"dataset_id: \n{DATASET_ID_TO_TRAIN}\n")
    fold = 0
    print(f"fold: \n{fold}\n")
    train(DATASET_ID_TO_TRAIN, fold, device, configuration)


if __name__ == '__main__':
    main()
