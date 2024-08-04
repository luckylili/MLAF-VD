import pandas as pd  
  
class CWELabeler:  
    def __init__(self):  
        self.cwe_to_label = {}  
        self.label_counter = 1  
  
    def create_cwe_label(self, row):  
        if row['vul'] == 0:  
            return 0  
        else:  
            cwe_id = row['CWE ID']  
            if cwe_id not in self.cwe_to_label:  
                self.cwe_to_label[cwe_id] = self.label_counter  
                self.label_counter += 1  
            return self.cwe_to_label[cwe_id]  
  
df = pd.read_csv('MSR_CVE_CWE_commit.csv')  
  
labeler = CWELabeler()  
  
df['cwe_label'] = df.apply(labeler.create_cwe_label, axis=1)  

print("CWE ID to cwe_label mapping:")  
print(labeler.cwe_to_label)  
  
df.to_csv('MSR_CVE_CWE_commit_cwelabel.csv', index=False)


