'''创建DSG（Dangerous Structure Graph）'''
import argparse
import csv
import json
import traceback
from types import SimpleNamespace
import numpy as np
import os
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt 
import re
import warnings
import torch
import sys
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification  
from tokenizers import Tokenizer
import random  
import copy  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if input_ids is not None:
            outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
        else:
            outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
        feature=outputs[:, 0, :]  
        return feature

config = RobertaConfig.from_pretrained("microsoft/codebert-base")
config.num_labels = 1
config.num_attention_heads=12
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config, ignore_mismatched_sizes=True) 
model = Model(model, config, tokenizer)
model = model.to(device)

code_batch_size=512 

def convert_examples_to_features(func, tokenizer):
    token_length = len(tokenizer.tokenize(str(func))) 
    code_tokens = tokenizer.tokenize(str(func))[:code_batch_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = code_batch_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids


type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}

type_map_dsg = [
'AdditiveExpression', 
'SizeofExpression', 
'InclusiveOrExpression', 
'BitAndExpression', 
'MultiplicativeExpression',
'AssignmentExpression', 
'RelationalExpression', 
'CallExpression', 
'CastExpression',
'OrExpression', 
'AndExpression', 
'EqualityExpression', 
'ConditionalExpression', 
'Expression', 
'UnaryOperationExpression',
'ShiftExpression', 
'ExclusiveOrExpression', 
'PostIncDecOperationExpression', 
'CFGErrorNode', 
'Symbol',
'PtrMemberAccess', 
'DoStatement', 
'GotoStatement', 
'IfStatement', 
'TryStatement',
'WhileStatement',
]

type_one_hot = np.eye(len(type_map))
edgeType_full = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11, 
    'IS_FUNCTION_OF_CFG': 12, 
}



edgeType_control_light = {
    'FLOWS_TO': 3,  
    'CONTROLS': 7,  
}

edgeType_cfg = {
    'FLOWS_TO': 3,  
    'REACHES': 6,
    'CONTROLS': 7,  
    'DOM': 9,
    'POST_DOM': 10,
}

edgeType_dfg = {
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
}

edgeType_ast = {
    'IS_AST_PARENT': 1,
}

edgeType_dsg = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
}





def checkVul(cFile):
    with open(cFile, 'r',encoding='utf-8') as f:
        fileString = f.read()
        return (1 if "BUFWRITE_COND_UNSAFE" in fileString or "BUFWRITE_TAUT_UNSAFE" in fileString else 0)


def inputGeneration(file_name, output_embedding_path, nodeCSV, edgeCSV, target, edge_type_map, cfg_only=False):
    gInput = dict()
    gInput["targets"] = list()
    gInput["targets"].append([target])
    
    eInput = dict()
    eInput["dsg_node_code_e1_n1"] = list()
    eInput["dsg_graph_e1_n1"] = list()
    
    eInput["dsg_node_features_e1_n1"] = list()
    
    with open(nodeCSV, 'r',encoding='utf-8') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')
        nodeMap = dict()
        allNodesCode = {}
        allNodes = {}
        node_idx = 0
        allNodesTypes = {}
        for idx, node in enumerate(nodes):
            nodeKey = node['key']
            node_type = node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()

            allNodesTypes[nodeKey]=node_type 
            allNodesCode[nodeKey] = node_content

            nodeMap[nodeKey] = node_idx
            node_idx += 1
        if node_idx == 0 :
            return None
 
        trueNodeMap = {}
        all_nodes_with_edges = [] 
        all_edges = []
        
        cfg_nodes_with_edges= [] 
        cfg_edges = []
        
        dfg_nodes_with_edges= [] 
        dfg_edges = []
        
        dsg_nodes_with_edges= [] 
        dsg_edges = []
        
        with open(edgeCSV, 'r',encoding='utf-8') as ec:
            reader = csv.DictReader(ec, delimiter='\t')
            for e in reader:
                start, end, eType = e["start"], e["end"], e["type"]
                if eType != "IS_FILE_OF":
                    if start in nodeMap and end in nodeMap and eType in edgeType_cfg:   
                        if start not in all_nodes_with_edges:
                            all_nodes_with_edges.append(start)
                        if end not in all_nodes_with_edges:
                            all_nodes_with_edges.append(end)
                            
                        if start not in cfg_nodes_with_edges:
                            cfg_nodes_with_edges.append(start)
                        if end not in cfg_nodes_with_edges:
                            cfg_nodes_with_edges.append(end)
                        edge = [start, edgeType_cfg[eType], end]

                        all_edges.append(edge)
                        cfg_edges.append(edge)
                    if start in nodeMap and end in nodeMap and eType in edgeType_dfg: 
                        if start not in all_nodes_with_edges:
                            all_nodes_with_edges.append(start)
                        if end not in all_nodes_with_edges:
                            all_nodes_with_edges.append(end)
                            
                        if start not in dfg_nodes_with_edges:
                            dfg_nodes_with_edges.append(start)
                        if end not in dfg_nodes_with_edges:
                            dfg_nodes_with_edges.append(end)
                        edge = [start, edgeType_dfg[eType], end]
                        all_edges.append(edge)
                        dfg_edges.append(edge)   

                    if start in nodeMap and end in nodeMap and eType in edgeType_dsg: 
                        if start not in all_nodes_with_edges:
                            all_nodes_with_edges.append(start)
                        if end not in all_nodes_with_edges:
                            all_nodes_with_edges.append(end)
                            
                        if start not in dsg_nodes_with_edges:
                            dsg_nodes_with_edges.append(start)
                        if end not in dsg_nodes_with_edges:
                            dsg_nodes_with_edges.append(end)
                        edge = [start, edgeType_dsg[eType], end]
                        all_edges.append(edge)
                        dsg_edges.append(edge)   
                                
         
        
        random.shuffle(dsg_nodes_with_edges, random.Random(random_seed).random)         
        
        all_nodes_with_edges = sorted(all_nodes_with_edges)
        if len(cfg_edges) == 0:
            return None
        if len(dfg_edges) == 0:
            return None

        
        if len(cfg_nodes_with_edges) > 200: 
            return None    
        if len(dfg_nodes_with_edges) > 500: 
            return None 
            
        dsg_edges_simplify = []  
        dsg_nodes_with_edge_simplify = []  
        valid_nodes = []  
  
        for node in dsg_nodes_with_edges:  
            if allNodesTypes[node] in type_map_dsg:  
                dsg_nodes_with_edge_simplify.append(node)  
                if node not in valid_nodes:  
                    valid_nodes.append(node)  
  
        for edge in dsg_edges:  
            start, edge_type, end = edge  
            if start in valid_nodes and end in valid_nodes:  
                if edge not in dsg_edges_simplify:  
                    dsg_edges_simplify.append(edge)  
            elif start in valid_nodes and end not in valid_nodes:  
                for existing_edge in dsg_edges:  
                    existing_start, existing_edge_type, existing_end = existing_edge  
                    if existing_start == end and existing_edge_type == edge_type and existing_end in valid_nodes:  
                        new_edge = [start, edge_type, existing_end]  
                        if new_edge not in dsg_edges_simplify:  
                            dsg_edges_simplify.append(new_edge) 

                
        dsg_nodes_with_edge_simplify_e1_n1 = copy.deepcopy(dsg_nodes_with_edge_simplify)
        dsg_edges_simplify_e1_n1 = copy.deepcopy(dsg_edges_simplify)  
        for i, node in enumerate(dsg_nodes_with_edge_simplify_e1_n1):
            trueNodeMap[node] = i
            eInput["dsg_node_code_e1_n1"].append(allNodesCode[node])         
            inputs = convert_examples_to_features(allNodesCode[node], tokenizer)      
            inputs_ids_one=torch.tensor(inputs)
            inputs_ids_one = inputs_ids_one.to(device)  
            inputs_ids=inputs_ids_one.unsqueeze(0) 
            outputs = model(input_ids=inputs_ids)
            if outputs.shape[1] > 0:  
                fNrp = outputs.squeeze().detach().cpu().numpy() 
            else:  
                fNrp = np.zeros(model.config.hidden_size)  
                print(fNrp)                
            node_feature=[]
            node_feature.extend(fNrp.tolist())
            allNodes[node]=node_feature         
            eInput["dsg_node_features_e1_n1"].append(allNodes[node])              
        for edge in dsg_edges_simplify_e1_n1:
            start, t, end = edge
            start = trueNodeMap[start]
            end = trueNodeMap[end]
            e = [start, t, end]
            eInput["dsg_graph_e1_n1"].append(e) 
            
        embedding_file=output_embedding_path+'/'+file_name
        if not os.path.exists(embedding_file):  
            os.makedirs(embedding_file)     
            
        with jsonlines.open(embedding_file+'/dsg_node_features_1_types.jsonlines', "w") as writer:
            features_point = { 
                    "dsg_node_code_e1_n1": eInput["dsg_node_code_e1_n1"],
                    "dsg_graph_e1_n1": eInput["dsg_graph_e1_n1"],                
            }
            writer.write(features_point)
        
        np.save(embedding_file+'/dsg_node_features_e1_n1_new.npy', eInput["dsg_node_features_e1_n1"])  
        
        eInput["dsg_node_code_e1_n1"] = []
        eInput["dsg_graph_e1_n1"] = []
    
        eInput["dsg_node_features_e1_n1"] = []
            
    return gInput
    

def unify_slices(list_of_list_of_slices):
    taken_slice = set()
    unique_slice_lines = []
    for list_of_slices in list_of_list_of_slices:
        for slice in list_of_slices:
            slice_id = str(slice)
            if slice_id not in taken_slice:
                unique_slice_lines.append(slice)
                taken_slice.add(slice_id)
    return unique_slice_lines
    pass



random_seed = 123456
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

import jsonlines
def main():
    from types import SimpleNamespace
    args = SimpleNamespace()
    data_dir = 'D:/[Experiment]/[source][Devign、LineVul、SySeVR、VulBERTa]/data-package/datasets/test'
    args.csv = f'{data_dir}/parsed/tmp_test_one/'
     
    output_path = 'D:/[Experiment]/[source][Devign、LineVul、SySeVR、VulBERTa]/data-package/datasets/test'
    args.output =output_path+f'/full_experiment_real_data_test/dsg_test_codes_nodes_edges.jsonlines'
    output_embedding_path=output_path+'/dsg_Embedding_test'
    
    json_file_path = data_dir + "/" + 'full_data_with_slices.json'
    data = json.load(open(json_file_path))
    print("loaded", len(data), "examples")
    v, nv = 0, 0
    
    skipped = 0

    print(data_dir)
    print(args.output)
    with jsonlines.open(args.output, "w") as writer:
        for didx, entry in enumerate(tqdm(data, initial=skipped, total=len(data) + skipped)):
            if "file_name" in entry:
                file_name = entry['file_name']
            else:
                file_name = entry['file_path'].split('/')[-1]
                
            nodes_path = os.path.join(args.csv, file_name, 'nodes.csv')
            edges_path = os.path.join(args.csv, file_name, 'edges.csv')
            label = int(entry['label'])
            if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
                continue
            linized_code = {}
            for ln, code in enumerate(entry['code'].split('\n')):
                linized_code[ln + 1] = code

            graph_input_full = None
            try:
                graph_input_full = inputGeneration(
                    file_name, output_embedding_path, nodes_path, edges_path, label, edgeType_full, False)
            except Exception:
                print(traceback.format_exc())
                pass

            if graph_input_full is None:
                continue
            if label == 1:
                v += 1
            else:
                nv += 1
            
            data_point = { 
                'id': didx,
                'file_name': file_name,
                'file_path': output_embedding_path+'/'+file_name,
                'code': entry['code'],
                'full_graph': graph_input_full,
                'label': int(entry['label'])
            }
            writer.write(data_point)
    
    print("Vulnerable:\t%d\n"
          "Non-Vul:\t%d\n"
          % \
          (v, nv))


if __name__ == '__main__':
    main()

