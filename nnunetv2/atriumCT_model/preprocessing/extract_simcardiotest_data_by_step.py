import pandas as pd
import os
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk


if __name__ == "__main__":

    src_dir = '/media/SimCardioTest/images_feb2022/simcardiotest_by_step/'
    nnunet_dir = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset006_LA_CT00_region_fenetrage/simcardiotest/'


    patients = [f for f in os.listdir(src_dir) if f[-4:]!='.csv']
    df = pd.DataFrame({'patient': patients})
    df.index.name='ID'
    print(df)
    df.to_csv('/media/SimCardioTest/images_feb2022/simcardiotest_by_step/patients_indexes.csv')

    maybe_mkdir_p(nnunet_dir)
    k=0
    for i in range(len(patients)):
        patient_path = os.path.join(src_dir, patients[i])
        print(patient_path)
        for j in range(20):
            print(os.path.join(patient_path, f'image_{j}.nii'))
            print(os.path.join(nnunet_dir, f'patient_{i}_step_{j}_{k:03}_0000.mha'))
            img = sitk.ReadImage(os.path.join(patient_path, f'image_{j}.nii'))
            sitk.WriteImage(img, os.path.join(nnunet_dir, f'patient_{i}_step_{j}_{k:03}_0000.mha'))
            k+=1
    print(df)
        