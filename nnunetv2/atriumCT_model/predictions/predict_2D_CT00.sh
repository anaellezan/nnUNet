#!/bin/bash
nnUNetv2_predict_from_modelfolder -i /media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset002_LA_CT00/val_data/fold_0/imagesTr/ -o /media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset002_LA_CT00/TEST_2D/ -m /media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset002_LA_CT00/nnUNetTrainer__nnUNetPlans__2d/ -f 0 --save_probabilities -chk checkpoint_best.pth