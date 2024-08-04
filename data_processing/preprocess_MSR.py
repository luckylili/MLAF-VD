import json
js_all=json.load(open('full_data_with_slices.json'))

'''
train_index=set()
valid_index=set()
test_index=set()

with open('train.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))
                    
with open('valid.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))
        
with open('test.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))

        
        
with open('train.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('valid.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in valid_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            
with open('test.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in test_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
'''           
with open('data.jsonlines','w') as f:
    for idx,js in enumerate(js_all):
        js['idx']=idx
        f.write(json.dumps(js)+'\n')
