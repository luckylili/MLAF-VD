import os  
  
def convert_newlines_and_encoding(directory):  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.c'):  
                file_path = os.path.join(root, file)  
                try:  
                    with open(file_path, 'r', encoding='gbk') as f:  
                        content = f.read()  
                except UnicodeDecodeError:  
                    with open(file_path, 'r', encoding='utf-8') as f:  
                        content = f.read()  
                new_content = content.replace('\r\n', '\n').replace('\r', '\n')  
                if new_content != content:  
                    with open(file_path, 'w', encoding='utf-8') as f:  
                        f.write(new_content)  
  
convert_newlines_and_encoding('raw_code')