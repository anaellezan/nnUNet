''' LAST VERSION FOR FINETUNING WITH MASKS.
Using pretrained model: Dataset002, training stopped at 500th epoch to avoid overfitting. 
Run training on Dataset004_LA_CT00_corrected_voted from images and LA binary mask resampled into images spaces.
'''

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

    # preprocessing
    print('Preprocessing...')
    preprocess(
        dataset_ids=[DATASET_ID_TO_TRAIN], configurations=CONFIGURATIONS,
    )


def train(dataset_id, fold, device, conf, pretrained_weights, plans_identifier,):

    run_training(
        dataset_name_or_id=dataset_id,
        configuration=conf,
        nb_folds=FOLDS_NUMBER,
        fold=fold,
        export_validation_probabilities=EXPORT_VALIDATION_PROBABILITY,
        device=device,
        num_epochs=1500,
        continue_training=False,
        pretrained_weights=pretrained_weights,
        plans_identifier=plans_identifier,
        val_with_best=True # /!\
    )


def main():

    device = torch.device('cuda')
    fold = 0
    configuration = '3d_fullres'
    pretrained_weights_path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_preprocessed/Dataset004_LA_CT00_corrected_voted/checkpoint_best_Dataset002_fold_0_3d_fullres_500_epochs.pth'
    plan = 'nnUNetPlans_Dataset002'
    print(f"configuration: \n{configuration}\n")
    print(f"dataset_id: \n{DATASET_ID_TO_TRAIN}\n")
    print(f"fold: \n{fold}\n")
    train(DATASET_ID_TO_TRAIN, fold, device, configuration, pretrained_weights_path, plan)



if __name__ == '__main__':
    main()
