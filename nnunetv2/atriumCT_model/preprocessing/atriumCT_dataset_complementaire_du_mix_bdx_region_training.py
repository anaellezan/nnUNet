from multiprocessing import Pool
from tqdm import tqdm
from os import listdir

import shutil

from batchgenerators.utilities.file_and_folder_operations import *


## FAIRE LE DISTINGO ENTRE RAW DONEES SOURCES ET LE RAW DE NNUNET 
if __name__ == "__main__":

    mix_dir = '/media/DATA/Toulouse/mix_donnes_predites_GT/Images/'
    bdx_dir = '/media/sharedata/atriumCT/corrected_data/GTmasks2'
    nnunet_dir = '/media/sharedata/atriumCT/atrium_nnunet/raw_data/Dataset006_LA_CT00_region_fenetrage/bordeaux_complementaire/'
    maybe_mkdir_p(nnunet_dir)
    
    mix = listdir(mix_dir)
    bdx = listdir(bdx_dir)
    bordeaux_complementaire = [f for f in mix if f not in bdx]

    all_cases = [f[:-4] for f in bordeaux_complementaire if f[0]!='.']
    all_cases.sort(key=int)
    all_cases
    print(f'\n{len(all_cases)} cases in the source dir.\n')
    
    
    for i in tqdm(range(len(all_cases))):
        # copy images
        print(all_cases[i])
        print(f'{mix_dir}{all_cases[i]}.mha')
        print(f'{nnunet_dir}bdx_complementaire_la_{all_cases[i]}_{i:03}_0000.mha')
        shutil.copy(f'{mix_dir}{all_cases[i]}.mha', f'{nnunet_dir}bdx_complementaire_la_{all_cases[i]}_{i:03}_0000.mha')

