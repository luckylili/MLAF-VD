import json  
import os  
  
json_file_path = 'vulnerables.json'  
  
with open(json_file_path, 'r', encoding='utf-8') as file:  
    data = json.load(file)  
  
if not os.path.exists('raw_code'):  
    os.makedirs('raw_code')  
  
for index, item in enumerate(data):  
    project_name = item['project']  
    code = item['code']  
      
    file_name = f"{project_name}_{index + 1}_1.c"  
    file_path = os.path.join('raw_code', file_name)  
      
    with open(file_path, 'w', encoding='utf-8') as file:  
        file.write(code.replace('\r\n', '\n'))  
  
print("写入完成。")