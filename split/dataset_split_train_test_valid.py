import pandas as pd  
import json  
  
df = pd.read_excel('Devign_0.1_123456.xlsx')  
file_name_to_split_map = dict(zip(df['folder_name'], df['split']))  
  
train_file = open('dataset_split/train_0.1_123456.jsonlines', 'w')  
valid_file = open('dataset_split/valid_0.1_123456.jsonlines', 'w')  
test_file = open('dataset_split/test_0.1_123456.jsonlines', 'w')  
  
output_files = {  
    'train': train_file,  
    'valid': valid_file,  
    'test': test_file  
}  
  
with open('full_experiment_real_data/cfg_dfg_astlp_data_0.1.jsonlines', 'r') as jsonlines_file:  
    for line in jsonlines_file:     
        data = json.loads(line)  
        file_name = data['file_name']  
        print(file_name)
        split = file_name_to_split_map.get(file_name, None)  
        if split:  
            output_files[split].write(line)  
  
for f in output_files.values():  
    f.close()