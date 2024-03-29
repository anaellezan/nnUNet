from multiprocessing import Pool
from tqdm import tqdm
from os import listdir

import shutil


## FAIRE LE DISTINGO ENTRE RAW DONEES SOURCES ET LE RAW DE NNUNET 
if __name__ == "__main__":

    source_dir = '/media/DATA/Toulouse/Scanner_filtered/'
    nnunet_dir = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset006_LA_CT00_region_fenetrage/toulouse/'


    all_cases = [f for f in listdir(source_dir) if f[0]!='.']
    all_cases.sort(key=int)
    all_cases
    print(f'\n{len(all_cases)} cases in the source dir.\n')
    
    
    for i in tqdm(range(len(all_cases))):
        # copy images
        print(all_cases[i])
        if all_cases[i] == '108':
            file = 'HALF 738ms 0.91s Coeur 0.5 CE.mha'
            print(file)
        else:
            files = [f for f in listdir(f'{source_dir}{all_cases[i]}') if f[-4:]=='.mha']
            if len(files) == 0:
                continue
            else :
                file = files[0]
                print(file)
        print(f'{source_dir}{all_cases[i]}/{file}')
        print(f'{nnunet_dir}toulouse_la_{all_cases[i]}_{i:03}_0000.mha')
        shutil.copy(f'{source_dir}{all_cases[i]}/{file}', f'{nnunet_dir}toulouse_la_{all_cases[i]}_{i:03}_0000.mha')

