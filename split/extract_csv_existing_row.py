import json  
import pandas as pd  
  
file_names_set = set()  
with open('linevul_split/all_dataset_linevul.jsonlines', 'r', encoding='utf-8') as file:  
    for line in file:  
        data = json.loads(line)  
        file_name = data['file_name']  
        number = file_name.split('_')[0]  
        file_names_set.add(number)  
  
df = pd.read_csv('MSR_CVE_CWE_commit_cwelabel.csv')  
filtered_df = df[df['index'].astype(str).isin(file_names_set)]  
  
filtered_df.to_csv('MSR_CVE_CWE_commit_cwelabel_linevul_jsonlines.csv', index=False)