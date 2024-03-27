import pandas as pd
from tqdm import tqdm
import os
import SimpleITK as sitk


pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


path = '/media/sharedata/atriumCT/atrium_nnunet/nnUNet_results/Dataset006_LA_CT00_region_fenetrage/nnUNetTrainer__nnUNetPlans_Dataset002__3d_fullres_classic_loss/fold_0/bordeaux_complementaire_prediction/'
files = [f for f in os.listdir( path ) if f[-4:]=='.mha']
print(len(files))


def main():

    ids = [int(file.split('_')[-2]) for file in files]
    df = pd.DataFrame({'ID':ids})
    print(df)

    for file in tqdm(files):
        origin_id = int(file.split('_')[-2])
        seg_ref = sitk.ReadImage(f'{path}{file}')
        auricle = ((seg_ref == 3)|(seg_ref == 4))
        seg_ref_auricule_arr = sitk.GetArrayFromImage(auricle)
        seg_ref_opa_defect = (seg_ref == 4)
        seg_ref_opa_defect_arr = sitk.GetArrayFromImage(seg_ref_opa_defect)


        vol_auricle_ref = seg_ref_auricule_arr.sum()
        vol_opa_ref = seg_ref_opa_defect_arr.sum()

        opa_percentage = seg_ref_opa_defect_arr.sum() / vol_auricle_ref * 100
        df.loc[df['ID']==origin_id,'Opa_percentage'] = opa_percentage

        spacing_0= seg_ref.GetSpacing()[0]
        spacing_1= seg_ref.GetSpacing()[1]
        spacing_2= seg_ref.GetSpacing()[2]

        df.loc[df['ID']==origin_id,'Spacing_ref_0'] = spacing_0
        df.loc[df['ID']==origin_id,'Spacing_ref_1'] = spacing_1
        df.loc[df['ID']==origin_id,'Spacing_ref_2'] = spacing_2

        df.loc[df['ID']==origin_id,'vol_auricle_ref'] = vol_auricle_ref
        df.loc[df['ID']==origin_id,'vol_opa_ref'] = vol_opa_ref

        df.loc[df['ID']==origin_id,'real_vol_auricle_ref'] = vol_auricle_ref * spacing_0 * spacing_1 * spacing_2
        df.loc[df['ID']==origin_id,'real_vol_opa_ref'] = vol_opa_ref * spacing_0 * spacing_1 * spacing_2

        print(origin_id)
        print(opa_percentage)
    
    # Thresholds define after the study on the Bordeaux dataset opa_correlation_study.ipynb
    df['thresholds_1481'] = (df['real_vol_opa_ref'] > 1481).astype(int)
    print(df)
    df.to_csv(f'{path}patients_history_percentage_opa_correlation_study_bordeaux_complementaire.csv', index=False)


if __name__ == '__main__':
    main()