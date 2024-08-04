import json  
import os
import pandas as pd  
  
results = pd.DataFrame(columns=['id', 'file_name', 'cfg_graph count', 'cfg_node_code count', 'dfg_graph count', 'dfg_node_code count', 'ast_lp count','label'])  
  
jsonlines_file_path = 'test_0.1_123456.jsonlines'  
  
new_rows = []  
  
with open(jsonlines_file_path, 'r') as file:  
    for line in file:  
        data = json.loads(line)  
          
        id = data['id']  
        file_name = data['file_name']  
        cfg_edges = data['full_graph']['cfg_graph']  
        cfg_nodes = data['full_graph']['cfg_node_code']  
        dfg_edges = data['full_graph']['dfg_graph']  
        dfg_nodes = data['full_graph']['dfg_node_code']  
        ast_lp_edges = data['full_graph']['ast_lp'] 
        label = data['label']        
          
        cfg_edges_count = len(cfg_edges)  
        cfg_nodes_count = len(cfg_nodes)  
        dfg_edges_count = len(dfg_edges)  
        dfg_nodes_count = len(dfg_nodes)  
        ast_lp_edges_count = len(ast_lp_edges)       
          
        new_row = {  
            'id': id,  
            'file_name': file_name,  
            'cfg_graph count': cfg_edges_count,  
            'cfg_node_code count': cfg_nodes_count,  
            'dfg_graph count': dfg_edges_count,  
            'dfg_node_code count': dfg_nodes_count,  
            'ast_lp count': ast_lp_edges_count
            'label': label            
        }  
        new_rows.append(new_row)  
  
new_rows_df = pd.DataFrame(new_rows)  
  
results = pd.concat([results, new_rows_df], ignore_index=True)  
 
output_file_path = jsonlines_file_path.replace("jsonlines", "xlsx")
results.to_excel(output_file_path, index=False)