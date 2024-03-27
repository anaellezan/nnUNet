import pandas as pd
from tqdm import tqdm
import os
import SimpleITK as sitk


pd.options.display.max_rows = 999
pd.options.display.max_columns = 999


dataset = 'Dataset004_LA_CT00_corrected_voted'
path = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset005_LA_CT00_corrected_voted_region/labelsTr/'
files = [f for f in os.listdir( path ) if f[-4:]=='.mha']
print(len(files))


# ref_label_files = ['/media/sharedata/atriumCT/corrected_data/GTlabels_with_opa_defect/' + f for f in files]
# print(len(ref_label_files))

def main():

    df = pd.read_csv(f'/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset004_LA_CT00_corrected_voted/prediction_3d_dataset002/patients_history.csv')

  
    for i in tqdm(range(len(files))):
        seg_ref = sitk.ReadImage(f'{path}{files[i]}')
        auricle = ((seg_ref == 2)|(seg_ref == 7))
        seg_ref_auricule_arr = sitk.GetArrayFromImage(auricle)
        seg_ref_opa_defect = (seg_ref == 7)
        seg_ref_opa_defect_arr = sitk.GetArrayFromImage(seg_ref_opa_defect)


        vol_auricle_ref = seg_ref_auricule_arr.sum()
        vol_opa_ref = seg_ref_opa_defect_arr.sum()

        opa_percentage = seg_ref_opa_defect_arr.sum() / vol_auricle_ref * 100
        origin_id = int(files[i][-11:-8].split('_')[-1])
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
    print(df)
    df.to_csv(f'{path}patients_history_percentage_opa_correlation_study.csv', index=False)


if __name__ == '__main__':
    main()