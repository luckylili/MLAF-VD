import pandas as pd

sample = False
df = pd.read_csv("../../../../datasets/MSR/MSR_data_cleaned.csv",low_memory=False,index_col=0, nrows=5 if sample else None)
df

def get_filename(row, version):
    return "_".join([str(row.name), row["project"], row["commit_id"], version, str(row["vul"])]) + ".c"
get_filename(df.iloc[0], "before")


import os
code_dir = "../../../../datasets/MSR/raw_code"
os.makedirs(code_dir, exist_ok=True)
def extract_code(row):
    if row["vul"]:
        versions = ["before", "after"]
    else:
        versions = ["before"]
    for version in versions:
        filename = str(get_filename(row, version)).replace('?w=1', '')
        print(filename)
        filepath = os.path.join(code_dir, filename)
        with open(filepath, "w",encoding='utf-8') as f:
            f.write(row["func_" + version])
df.apply(extract_code, axis=1)

code_dir = "../../../../datasets/MSR/raw_code"
def extract_filename(row):
    if row["vul"]:
        versions = ["before", "after"]
    else:
        versions = ["before"]
    for version in versions:
        yield get_filename(row, version)
with open("../../../../datasets/MSR/files.txt", "w",encoding='utf-8') as f:
    for i, row in df.iterrows():
        for filename in extract_filename(row):
            f.write(filename + "\n")
