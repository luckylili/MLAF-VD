import os  
import json  
  
def create_jsonlines(folder_path, output_file):  
    with open(output_file, 'w') as outfile:  
        for filename in os.listdir(folder_path):  
            if filename.endswith('_1.c') or filename.endswith('_0.c'):  
                file_path = os.path.join(folder_path, filename)  
                with open(file_path, 'r',encoding='utf-8') as file:  
                    content = file.read()  
                    target = 1 if filename.endswith('_1.c') else 0  
                    sample_dict = {'target': target, 'func': content}  
                    json.dump(sample_dict, outfile)  
                    outfile.write('\n')  
  
folder_path = 'raw_code'  
output_file = 'function_reveal.jsonlines'  
create_jsonlines(folder_path, output_file)