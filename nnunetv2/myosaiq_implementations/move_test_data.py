import pandas as pd
from os import listdir, remove
from os.path import isfile, join, exists
import shutil

test_path = '/beegfs/azanella/data_challenge/nnUNet/nnUNet_raw/imagesTs/'

d8_test = [(test_path + f) for f in listdir(test_path) if 'D8_' in f]
m1_test = [(test_path + f) for f in listdir(test_path) if 'M1_' in f]
m12_test = [(test_path + f) for f in listdir(test_path) if 'M12_' in f]
print(f'Here is the D8 test dataset of length {len(d8_test)}: \n\n{d8_test}\n\n')
print(f'Here is the M1 test dataset of length {len(m1_test)}: \n\n{m1_test}\n\n')
print(f'Here is the M12 test dataset of length {len(m12_test)}: \n\n{m12_test}\n\n')


for f in d8_test:
    shutil.copyfile(f, f.replace('imagesTs', 'training/Dataset010_D8/imagesTs'))

for f in m1_test:
    shutil.copyfile(f, f.replace('imagesTs', 'training/Dataset011_M1/imagesTs'))

for f in m12_test:
    shutil.copyfile(f, f.replace('imagesTs', 'training/Dataset012_M12/imagesTs'))