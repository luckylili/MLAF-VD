import csv  
import json  
import sys  

maxInt = sys.maxsize  
  
while True:  
    try:  
        csv.field_size_limit(maxInt)  
        break  
    except OverflowError:  
        maxInt = int(maxInt/10) 

print(maxInt)        
  
input_csv = 'MSR_data_cleaned.csv'  
output_jsonl = 'MSR_data_cleaned.jsonlines'  
  
with open(input_csv, 'r', encoding='utf-8', newline='') as csvfile, open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:  
    csvreader = csv.DictReader(csvfile)  
    for row in csvreader:  
        jsonlfile.write(json.dumps(row, ensure_ascii=False) + '\n')
