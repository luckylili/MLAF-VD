
import tqdm
import traceback
import json
import re
all_data = []

import sys
data_dir = f"{sys.argv[1]}"

split_dir = f'{data_dir}/raw_code/'

import os
files = os.listdir(f"{data_dir}/raw_code")

import csv 

def read_file(path):
    with open(path,encoding='utf-8') as f:
        lines = f.readlines()
        return ' '.join(lines)
    

    
for i, file_name  in tqdm.tqdm(enumerate(files), total=len(files), desc="process files"):
    label = file_name.strip()[:-2].split('_')[-1]
    code_text = read_file(split_dir + file_name.strip())
    

    data_instance = {
        'file_path': split_dir + file_name.strip(),
        'code' : code_text,
        'label': int(label)
    }
    all_data.append(data_instance)
    

len(all_data)


output_file = open(f'{data_dir}/full_data_with_slices.json', 'w')
json.dump(all_data, output_file)
output_file.close()




print(len(all_data))






