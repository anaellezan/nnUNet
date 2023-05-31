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
DATASET_IDS = [1, 2, 3]
CONFIGURATIONS = ['2d', '3d_fullres', '3d_lowres']

# /!\
FOLDS_NUMBER = 10
DATASET_ID_TO_TRAIN = 1
CONFIGURATION_TO_TRAIN = '3d_fullres'
EXPORT_VALIDATION_PROBABILITY = True



def plan_and_preprocess():
    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(dataset_ids=[10], check_dataset_integrity=True)

    # experiment planning
    print('Experiment planning...')
    plan_experiments(dataset_ids=[10],)
    # overwrite_plans_name: Optional[str] = None)

    # preprocessing
    print('Preprocessing...')
    preprocess(
        dataset_ids=[10], configurations=CONFIGURATIONS,
    )


def train(dataset_id, fold, device):

    # # multithreading in torch doesn't help nnU-Net if run on GPU
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    # device = torch.device('cuda')

    run_training(
        dataset_name_or_id=dataset_id,
        configuration=CONFIGURATION_TO_TRAIN,
        nb_folds=FOLDS_NUMBER,
        fold=fold,
        export_validation_probabilities=EXPORT_VALIDATION_PROBABILITY,
        device=device,
        num_epochs=200
    )
    #  pretrained_weights: Optional[str] = None,
    #  continue_training: bool = False,
    #  only_run_validation: bool = False,
    #  disable_checkpointing: bool = False,

    # print(torch.cuda.memory_allocated(device))
    # gc.collect()
    # torch.cuda.empty_cache()


def main():
    plan_and_preprocess()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda')

    configuration = '3d_fullres'
    dataset_id = 10

    # for configuration in CONFIGURATIONS:
    print(f"configuration: \n{configuration}\n")
        # for dataset_id in DATASET_IDS:
    print(f"dataset_id: \n{dataset_id}\n")
    for fold in range(FOLDS_NUMBER):
        print(f"fold: \n{fold}\n")
        train(dataset_id, fold, device)



if __name__ == '__main__':
    main()
