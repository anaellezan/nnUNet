import gc
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess,
)

import torch.cuda

from nnunetv2.run.run_training import run_training

# Run with CUDA_VISIBLE_DEVICES=X to specify GPU

# /!\
DATASET_IDS = [4, 5, 6]
CONFIGURATIONS = ['2d', '3d_fullres', '3d_lowres']

# /!\
FOLDS_NUMBER = 5
DATASET_ID_TO_TRAIN = 1
CONFIGURATION_TO_TRAIN = '3d_fullres'
EXPORT_VALIDATION_PROBABILITY = True


def plan_and_preprocess():
    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(dataset_ids=DATASET_IDS, check_dataset_integrity=True)

    # experiment planning
    print('Experiment planning...')
    plan_experiments(dataset_ids=DATASET_IDS,)
    # overwrite_plans_name: Optional[str] = None)

    # preprocessing
    print('Preprocessing...')
    preprocess(
        dataset_ids=DATASET_IDS, configurations=CONFIGURATIONS,
    )


def train(dataset_id, fold, device):

    run_training(
        dataset_name_or_id=dataset_id,
        configuration=CONFIGURATION_TO_TRAIN,
        fold=fold,
        export_validation_probabilities=EXPORT_VALIDATION_PROBABILITY,
        device=device,
        num_epochs=200
    )

    print(torch.cuda.memory_allocated(device))
    gc.collect()
    torch.cuda.empty_cache()
    #  pretrained_weights: Optional[str] = None,
    #  continue_training: bool = False,
    #  only_run_validation: bool = False,
    #  disable_checkpointing: bool = False,


def main():

    plan_and_preprocess()
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    # multithreading in torch doesn't help nnU-Net if run on GPU
    device = torch.device('cuda')
    print(torch.cuda.memory_allocated(device))
    

    for dataset_id in DATASET_IDS:
        print(f"dataset_id: \n{dataset_id}\n")
        for fold in range(0, FOLDS_NUMBER):
            print(f"fold: \n{fold}\n")
            train(dataset_id, fold, device)
            # gc.collect()
            # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
