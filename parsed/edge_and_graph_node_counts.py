import os  
import pandas as pd  
from collections import defaultdict  
from openpyxl import Workbook  
from tqdm import tqdm
  
wb = Workbook()  
ws = wb.active  
ws.title = "节点统计"  
  
columns = ["文件夹名称"]  
edge_types = ['IS_AST_PARENT', 'IS_CLASS_OF', 'FLOWS_TO', 'DEF', 'USE', 'REACHES', 'CONTROLS', 'DECLARES', 'DOM', 'POST_DOM', 'IS_FUNCTION_OF_AST', 'IS_FUNCTION_OF_CFG']  
for edge_type in edge_types:  
    columns.append(f"{edge_type}_节点数量")  
columns.extend(["CFG节点数量", "DFG节点数量", "AST节点数量"])  
ws.append(columns)  
  
edge_type_to_category = {  
    'FLOWS_TO': 'CFG',  
    'REACHES': 'CFG',  
    'CONTROLS': 'CFG',  
    'DOM': 'CFG',  
    'POST_DOM': 'CFG',  
    'DEF': 'DFG',  
    'USE': 'DFG',  
    'IS_AST_PARENT': 'AST',  
    'IS_CLASS_OF': '',  
    'DECLARES': '',  
    'IS_FUNCTION_OF_AST': '',  
    'IS_FUNCTION_OF_CFG': '',  
}  

root_dir = "tmp"  
  
for subdir, dirs, files in os.walk(root_dir):  
    dirs = tqdm(dirs, desc="遍历子文件夹", leave=False)  
    for dir in dirs:  
        dir_path = os.path.join(subdir, dir)  
        csv_path = os.path.join(dir_path, "edges.csv")  
          
        if os.path.exists(csv_path): 
            df = pd.read_csv(csv_path, sep='\t')  
              
            edge_counts = defaultdict(set)  
            cfg_nodes = set()  
            dfg_nodes = set()  
            ast_nodes = set()  
              
            for index, row in df.iterrows():  
                start_node = str(row['start'])  
                end_node = str(row['end'])   
                edge_type = row['type']  
 
                edge_counts[edge_type].add(start_node)  
                edge_counts[edge_type].add(end_node)  
                  
                category = edge_type_to_category.get(edge_type, '')  
                is_cfg = category == 'CFG' or edge_type == 'REACHES'  
                is_dfg = category == 'DFG' or edge_type == 'REACHES'  
                if is_cfg:  
                    cfg_nodes.add(start_node)  
                    cfg_nodes.add(end_node)  
                if is_dfg:  
                    dfg_nodes.add(start_node)  
                    dfg_nodes.add(end_node)  
                if category == 'AST':  
                    ast_nodes.add(start_node)  
                    ast_nodes.add(end_node)  
              
            row_data = [dir]  
            for edge_type in edge_types:  
                row_data.append(len(edge_counts[edge_type]))  
            row_data.append(len(cfg_nodes))  
            row_data.append(len(dfg_nodes))  
            row_data.append(len(ast_nodes))  
              
            ws.append(row_data)   
              
            edge_counts.clear()  
            cfg_nodes.clear()  
            dfg_nodes.clear()  
            ast_nodes.clear()  
  
excel_path = "Devign_edge_and_graph_nodes_count.xlsx"
wb.save(excel_path)  
print(f"节点统计已保存到 {excel_path}")