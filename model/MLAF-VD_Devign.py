'''[MLAF-VD]'''
import argparse
import logging
import torch  
import json
import os
import pickle
import random
import random as python_random
import numpy as np
import pandas as pd
from collections import defaultdict
import torch.nn as nn  
from torch.nn import CrossEntropyLoss, MultiheadAttention  
import torch.nn.functional as F 
import torch.nn.utils.rnn as rnn_utils 
import torch.nn.init as init  
from transformers import AdamW,get_linear_schedule_with_warmup,RobertaModel, RobertaConfig, RobertaTokenizer 
from torch_geometric.nn import GATConv, GlobalAttention 
from torch_geometric.utils import subgraph
from torch.utils.data import Dataset, DataLoader,SequentialSampler, RandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
from transformers import RobertaForSequenceClassification

logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    python_random.seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

  
class GlobalAttention(nn.Module):  
    def __init__(self, gate_nn, out_features):  
        super(GlobalAttention, self).__init__()  
        self.gate_nn = gate_nn  
        self.out_features = out_features  
        self.softmax = nn.Softmax(dim=0) 
  
    def forward(self, x, node_index): 
        gate_input = self.gate_nn(x)  
        attention_scores = gate_input.squeeze(1)  
        attention_weights = self.softmax(attention_scores)  
        aggregated_features = torch.mm(x.t(), attention_weights.unsqueeze(1)).t()  
        return aggregated_features.squeeze(0)  
  
class GATNetwork(nn.Module):  
    def __init__(self, in_channels, out_channels, num_heads, concat=True, dropout=0.3):  
        super(GATNetwork, self).__init__()
        self.convs = nn.ModuleList([        
            GATConv(in_channels=in_channels if i == 0 else out_channels * num_heads if concat else out_channels, 
                             out_channels=out_channels,
                             heads=num_heads,  
                             dropout=dropout,  
                             concat=concat)  
            for i in range(2)
        ])
        in_features = num_heads * out_channels if concat else out_channels
        
        self.global_attention = GlobalAttention(  
            gate_nn=nn.Sequential(  
                nn.Linear(in_features, out_channels // 2), 
                nn.LeakyReLU(negative_slope=0.001),  
                nn.Linear(out_channels // 2, 1)  
            ),  
            out_features=out_channels if concat else out_channels * num_heads  
        )   
  
    def forward(self, x, edge_index, edge_features,*args):   
        for i, conv in enumerate(self.convs):  
            x = F.elu(conv(x, edge_index, edge_features))  
            if i < len(self.convs) - 1 and hasattr(conv, 'concat') and not conv.concat:  
                x = F.dropout(x, p=0.3, training=self.training) 
        
        if hasattr(self.convs[-1], 'concat') and self.convs[-1].concat:  
            x = x.view(x.size(0), -1, self.convs[-1].heads, self.convs[-1].out_channels)  
            x = x.mean(dim=1) if not self.convs[-1].concat else x.view(x.size(0), -1)         
        else:
             x = x.mean(dim=1)  

                
        num_nodes = x.size(0)  
        node_index = torch.arange(num_nodes, device=x.device)   
       
        aggregated_feature = self.global_attention(x, node_index) 
        return aggregated_feature
        
class ResidualBlock(nn.Module):  
    def __init__(self, in_features, out_features, dropout_prob=0.1):  
        super(ResidualBlock, self).__init__()  
        self.fc1 = nn.Linear(in_features, out_features)  
        init.kaiming_normal_(self.fc1.weight)  
        self.bn1 = nn.BatchNorm1d(out_features)  
        self.relu = nn.ReLU()  
        self.dropout = nn.Dropout(dropout_prob)
          
        self.downsample = None  
        if in_features != out_features:  
            self.downsample = nn.Linear(in_features, out_features) 
            init.kaiming_normal_(self.downsample.weight)  
            
    def forward(self, x):  
        out = self.fc1(x)   
        out = self.bn1(out)  
        out = self.relu(out)  
        out = self.dropout(out)  
  
        if self.downsample is not None:  
            x = self.downsample(x)  
  
        out += x  
        out = self.relu(out)  
        return out                   
  
class CombinedClassifier(nn.Module):  
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_prob=0.1):  
        super(CombinedClassifier, self).__init__()  
        layers = []  
        prev_dim = input_dim  
          
        layers.append(nn.Dropout(dropout_prob))  
        layer = nn.Linear(prev_dim, hidden_dims[0])  
        layers.append(layer)  
        layers.append(nn.BatchNorm1d(hidden_dims[0]))  
        init.kaiming_normal_(layer.weight)  
        layers.append(nn.Tanh()) 
        layers.append(nn.Dropout(dropout_prob))  
        prev_dim = hidden_dims[0]  
  
        for i in range(1, len(hidden_dims)):  
            layers.append(ResidualBlock(prev_dim, hidden_dims[i], dropout_prob=dropout_prob))  
            prev_dim = hidden_dims[i]  
  
        layers.append(nn.Linear(prev_dim, output_dim))  
          
        self.layers = nn.Sequential(*layers) 

    def forward(self, x):
        x = self.layers(x)  
        return x
  
class ExtendedModel(nn.Module):  
    def __init__(self, encoder, roberta_config, gat_heads, gat_in_channels_cfg, gat_in_channels_dfg, gat_in_channels_dsg, gat_out_channels, hidden_dims, output_dim):  
        super(ExtendedModel, self).__init__()  
        self.encoder = encoder
        self.roberta = RobertaForSequenceClassification(roberta_config)        
               
        self.gat_network_cfg = GATNetwork(gat_in_channels_cfg, gat_out_channels, gat_heads, concat=True)  
        self.gat_network_dfg = GATNetwork(gat_in_channels_dfg, gat_out_channels, gat_heads, concat=True)
        
        self.gat_network_dsg = GATNetwork(gat_in_channels_dsg, gat_out_channels, gat_heads, concat=True)  
        
        self.fc_cfg = nn.Linear(gat_out_channels * gat_heads, gat_out_channels * gat_heads)  
        self.fc_dfg = nn.Linear(gat_out_channels * gat_heads, gat_out_channels * gat_heads)  
        self.fc_dsg = nn.Linear(gat_out_channels * gat_heads, gat_out_channels * gat_heads)  
        
        nn.init.kaiming_normal_(self.fc_cfg.weight, nonlinearity='relu')  
        nn.init.constant_(self.fc_cfg.bias, 0)  
        nn.init.kaiming_normal_(self.fc_dfg.weight, nonlinearity='relu')  
        nn.init.constant_(self.fc_dfg.bias, 0)  
        nn.init.kaiming_normal_(self.fc_dsg.weight, nonlinearity='relu')  
        nn.init.constant_(self.fc_dsg.bias, 0) 
        
        combined_features_dim = roberta_config.hidden_size + gat_out_channels * gat_heads* 3  
        self.classifier = CombinedClassifier(combined_features_dim, hidden_dims, output_dim)  
        
      
    def forward(self, input_ids, node_feats_cfg, edge_index_cfg, edge_type_cfg, node_feats_dfg, edge_index_dfg, edge_type_dfg, node_feats_dsg, edge_index_dsg, edge_type_dsg, labels=None,output_attentions=False):  
        input_ids = input_ids.squeeze(1)
        roberta_outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1),output_attentions=False)[0] 
        roberta_features=roberta_outputs[:, 0, :]
        

        batch_size = input_ids.size(0)  
        gat_features_cfg_list = []  
        gat_features_dfg_list = []
        gat_features_dsg_list = []

         
        for b in range(batch_size):  
            node_feats_cfg_b = node_feats_cfg[b]
            edge_index_cfg_b = edge_index_cfg[b]
            edge_type_cfg_b = edge_type_cfg[b]
            
            node_feats_dfg_b = node_feats_dfg[b]    
            edge_index_dfg_b = edge_index_dfg[b] 
            edge_type_dfg_b = edge_type_dfg[b]
            
            node_feats_dsg_b = node_feats_dsg[b]    
            edge_index_dsg_b = edge_index_dsg[b] 
            edge_type_dsg_b = edge_type_dsg[b]
            
            gat_features_cfg_b = self.gat_network_cfg(node_feats_cfg_b, edge_index_cfg_b, edge_type_cfg_b)  
            gat_features_dfg_b = self.gat_network_dfg(node_feats_dfg_b, edge_index_dfg_b, edge_type_dfg_b)  
            
            gat_features_dsg_b = self.gat_network_dsg(node_feats_dsg_b, edge_index_dsg_b, edge_type_dsg_b)  
  
            gat_features_cfg_list.append(gat_features_cfg_b.unsqueeze(0))  
            gat_features_dfg_list.append(gat_features_dfg_b.unsqueeze(0))  
            
            gat_features_dsg_list.append(gat_features_dsg_b.unsqueeze(0))  
  
        gat_features_cfg = torch.cat(gat_features_cfg_list, dim=0)  
        gat_features_dfg = torch.cat(gat_features_dfg_list, dim=0) 
        
        gat_features_dsg = torch.cat(gat_features_dsg_list, dim=0) 
        
        gat_features_cfg = self.fc_cfg(gat_features_cfg)  
        gat_features_dfg = self.fc_dfg(gat_features_dfg)  
        gat_features_dsg = self.fc_dsg(gat_features_dsg) 

          

        combined_features = torch.cat([roberta_features, gat_features_cfg, gat_features_dfg, gat_features_dsg], dim=1) 
        logits = self.classifier(combined_features)  
   

        prob = torch.softmax(logits, dim=-1)

        if labels is not None:  
            loss_fct = CrossEntropyLoss()  
            labels=labels.squeeze(dim=1)    
            loss = loss_fct(logits, labels)
            
            return loss, prob  
        else:  
            return prob  

  
class CodeGraphDataset(Dataset):  
    def __init__(self, tokenizer, args, file_type):
        self.tokenizer = tokenizer  
        self.args = args  
        if file_type == "train":
            self.file_path = args.train_data_file
        elif file_type == "eval":
            self.file_path = args.eval_data_file
        elif file_type == "test":
            self.file_path = args.test_data_file
        else:  
            raise ValueError(f"Invalid file_type: {file_type}")
        self.samples,self.label_to_indices= self._load_samples()  
        self.feature_file_dir=args.feature_file_dir 
        self.dsg_feature_file_dir=args.dsg_feature_file_dir        
  
    def _load_samples(self):  
        samples = []  
        label_to_indices = defaultdict(list)
        df = pd.read_json(self.file_path, lines=True, encoding='utf-8')  
        for index, row in df.iterrows():  
            sample = row.to_dict()  
            label = sample['label']  
            samples.append(sample)  
            label_to_indices[label].append(len(samples) - 1)  
  
        return samples, label_to_indices

    def __len__(self):  
        return len(self.samples)  
  
    def __getitem__(self, idx):  
        while True:
            sample = self.samples[idx]  
            code = sample['code'] 
            file_name = sample['file_name']
   
            input_ids = [self._text_to_token_ids(code)]
            input_ids = torch.tensor(input_ids)        
          
            cfg_node_feats = torch.tensor(np.load(self.feature_file_dir+'/'+file_name+'/cfg_node_features.npy').tolist())
            cfg_edge_index,cfg_edge_type = self._build_edge_index_and_features(sample['full_graph']['cfg_graph']) 
        
          
            dfg_node_feats = torch.tensor(np.load(self.feature_file_dir+'/'+file_name+'/dfg_node_features.npy').tolist())
            dfg_edge_index,dfg_edge_type = self._build_edge_index_and_features(sample['full_graph']['dfg_graph'])  
            
            dsg_node_feats = torch.tensor(np.load(self.dsg_feature_file_dir+'/'+file_name+'/dsg_node_features_e1_n1_new.npy').tolist())
            dsg_edge = pd.read_json(self.dsg_feature_file_dir+'/'+file_name+'/dsg_node_features_1_types.jsonlines', lines=True).iloc[0]
            dsg_edge_index,dsg_edge_type = self._build_edge_index_and_features(dsg_edge['dsg_graph_e1_n1'])  
            

            label = torch.tensor([sample['label']])  
        
            empty_tensor_found = False  
            if not input_ids.numel() :  
                empty_tensor_found = True  
            if cfg_node_feats.numel() == 0 or cfg_edge_index.numel() == 0 or cfg_edge_type.numel() == 0:  
                empty_tensor_found = True  
            if dfg_node_feats.numel() == 0 or dfg_edge_index.numel() == 0 or dfg_edge_type.numel() == 0:  
                empty_tensor_found = True 
            if dsg_node_feats.numel() == 0 or dsg_edge_index.numel() == 0 or dsg_edge_type.numel() == 0:  
                empty_tensor_found = True  
                          
            if not empty_tensor_found:  
                break  
              
            random.seed(123456)
            possible_indices = self.label_to_indices[sample['label']]  
            idx = random.choice(possible_indices)
    
        return (input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, label)  
  
    def _build_edge_index_and_features(self, graph):  
        edges = []  
        edge_features = []
        for edge in graph:  
            edges.append([edge[0], edge[2]])  
            edge_features.append(edge[1])  
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
        edge_features = torch.tensor(edge_features, dtype=torch.long).contiguous()   
        return edge_index, edge_features 
  
    def _text_to_token_ids(self, text):  
        code_tokens = self.tokenizer.tokenize(str(text))[:self.args.block_size - 2]  
        source_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]  
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)  
          
        padding_length = self.args.block_size - len(source_ids)  
        source_ids += [self.tokenizer.pad_token_id] * padding_length  
          
        return source_ids
        
def custom_collate(batch):  
    input_ids, cfg_node_feats, cfg_edge_indexes, cfg_edge_types, dfg_node_feats, dfg_edge_indexes, dfg_edge_types, dsg_node_feats, dsg_edge_indexes, dsg_edge_types, labels = zip(*batch)  
    stacked_input_ids = torch.stack(input_ids, dim=0)  
  
    max_nodes_cfg = max(feat.size(0) for feat in cfg_node_feats)  
    
    padded_cfg_node_feats = []  
    for feat in cfg_node_feats:  
        pad_size = max_nodes_cfg - feat.size(0)  
        if pad_size > 0:  
            pad = torch.zeros(pad_size, feat.size(1), dtype=feat.dtype, device=feat.device)  
            feat = torch.cat((feat, pad), dim=0)  
        padded_cfg_node_feats.append(feat)  
    stacked_cfg_node_feats = torch.stack(padded_cfg_node_feats, dim=0)
    
    max_nodes_dfg = max(feat.size(0) for feat in dfg_node_feats) 
    
    padded_dfg_node_feats = []  
    for feat in dfg_node_feats:  
        pad_size = max_nodes_dfg - feat.size(0)  
        if pad_size > 0:  
            pad = torch.zeros(pad_size, feat.size(1), dtype=feat.dtype, device=feat.device)  
            feat = torch.cat((feat, pad), dim=0)  
        padded_dfg_node_feats.append(feat)  
    stacked_dfg_node_feats = torch.stack(padded_dfg_node_feats, dim=0)
    
    max_nodes_dsg = max(feat.size(0) for feat in dsg_node_feats) 
    
    padded_dsg_node_feats = []  
    for feat in dsg_node_feats:  
        pad_size = max_nodes_dsg - feat.size(0)  
        if pad_size > 0:  
            pad = torch.zeros(pad_size, feat.size(1), dtype=feat.dtype, device=feat.device)  
            feat = torch.cat((feat, pad), dim=0)  
        padded_dsg_node_feats.append(feat)  
    stacked_dsg_node_feats = torch.stack(padded_dsg_node_feats, dim=0)
  
    list_cfg_edge_indexes = list(cfg_edge_indexes)    
    list_dfg_edge_indexes = list(dfg_edge_indexes) 
    
    list_dsg_edge_indexes = list(dsg_edge_indexes) 

    list_cfg_edge_types = list(cfg_edge_types)
    list_dfg_edge_types = list(dfg_edge_types) 

    list_dsg_edge_types = list(dsg_edge_types)    

    max_edges_cfg = max(edge_index.size(1) for edge_index in list_cfg_edge_indexes)  
    max_edges_dfg = max(edge_index.size(1) for edge_index in list_dfg_edge_indexes) 

    max_edges_dsg = max(edge_index.size(1) for edge_index in list_dsg_edge_indexes)     
  
    padded_cfg_edge_indexes = []  
    padded_cfg_edge_types = [] 
    for edge_index, edge_type in zip(list_cfg_edge_indexes, list_cfg_edge_types):  
        pad_size = max_edges_cfg - edge_index.size(1)  
        if pad_size > 0:  
            pad = torch.full((edge_index.size(0), pad_size), -1, dtype=edge_index.dtype, device=edge_index.device)
            edge_index = torch.cat((edge_index, pad), dim=1) 
            edge_type_pad = torch.full((pad_size,), 0, dtype=edge_type.dtype, device=edge_type.device)  
            edge_type = torch.cat((edge_type, edge_type_pad), dim=0)  
        padded_cfg_edge_indexes.append(edge_index)  
        padded_cfg_edge_types.append(edge_type) 
    stacked_cfg_edge_indexes = torch.stack(padded_cfg_edge_indexes, dim=0)  
    stacked_cfg_edge_types = torch.stack(padded_cfg_edge_types, dim=0)   
  
    padded_dfg_edge_indexes = []  
    padded_dfg_edge_types = []
    for edge_index, edge_type in zip(list_dfg_edge_indexes, list_dfg_edge_types):   
        pad_size = max_edges_dfg - edge_index.size(1)  
        if pad_size > 0: 
            pad = torch.full((edge_index.size(0), pad_size), -1, dtype=edge_index.dtype, device=edge_index.device)  
            edge_index = torch.cat((edge_index, pad), dim=1)  
            edge_type_pad = torch.full((pad_size,), 0, dtype=edge_type.dtype, device=edge_type.device)  
            edge_type = torch.cat((edge_type, edge_type_pad), dim=0) 
        padded_dfg_edge_indexes.append(edge_index)
        padded_dfg_edge_types.append(edge_type)          
    stacked_dfg_edge_indexes = torch.stack(padded_dfg_edge_indexes, dim=0)  
    stacked_dfg_edge_types = torch.stack(padded_dfg_edge_types, dim=0)    

    padded_dsg_edge_indexes = []  
    padded_dsg_edge_types = []
    for edge_index, edge_type in zip(list_dsg_edge_indexes, list_dsg_edge_types):   
        pad_size = max_edges_dsg - edge_index.size(1)  
        if pad_size > 0: 
            pad = torch.full((edge_index.size(0), pad_size), -1, dtype=edge_index.dtype, device=edge_index.device)  
            edge_index = torch.cat((edge_index, pad), dim=1)  
            edge_type_pad = torch.full((pad_size,), 0, dtype=edge_type.dtype, device=edge_type.device)  
            edge_type = torch.cat((edge_type, edge_type_pad), dim=0) 
        padded_dsg_edge_indexes.append(edge_index)
        padded_dsg_edge_types.append(edge_type)          
    stacked_dsg_edge_indexes = torch.stack(padded_dsg_edge_indexes, dim=0)  
    stacked_dsg_edge_types = torch.stack(padded_dsg_edge_types, dim=0)    

    stacked_labels = torch.stack(labels, dim=0)  
  
    return (stacked_input_ids, stacked_cfg_node_feats,  
            stacked_cfg_edge_indexes, stacked_cfg_edge_types, stacked_dfg_node_feats,  
            stacked_dfg_edge_indexes, stacked_dfg_edge_types, stacked_dsg_node_feats,  
            stacked_dsg_edge_indexes, stacked_dsg_edge_types, stacked_labels)
   

def train(args, train_dataset, model, tokenizer, eval_dataset):   
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0,pin_memory=True,collate_fn=custom_collate,drop_last=True)
    
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
                                                                                              
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0
    
    model.zero_grad()
    
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, labels = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)    

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss  

            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

            if step % args.gradient_accumulation_steps == 0:   
                optimizer.step()
                optimizer.zero_grad()      
                scheduler.step()  
                global_step += 1
                output_flag=True          

                if global_step % args.save_steps == 0:   
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)   
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)      

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)                        
        model_to_save = model.module if hasattr(model,'module') else model
        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)                        

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0,pin_memory=True,collate_fn=custom_collate,drop_last=True)  
    
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
        
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()  
    logits=[]  
    y_trues=[]
    for batch in tqdm(eval_dataloader, desc="evaluate eval"):
        input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, labels = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    print('【eval_loss】:',eval_loss)    
    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:,1]>best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds,average='binary', zero_division=0)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "eval_acc": float(acc),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result
    
def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0, pin_memory=True, collate_fn=custom_collate, drop_last=True)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(test_dataloader, desc="evaluate test"):
        input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, labels = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids, cfg_node_feats, cfg_edge_index, cfg_edge_type, dfg_node_feats, dfg_edge_index, dfg_edge_type, dsg_node_feats, dsg_edge_index, dsg_edge_type, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    print('【test_loss】:',eval_loss) 
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)   
    f1 = f1_score(y_trues, y_preds)             
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold":best_threshold,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    
    if args.write_raw_preds:
        logger.info('@@@@@')
        write_raw_preds_csv(args, y_preds)
        
def write_raw_preds_csv(args, y_preds):
    print('+++++')
    print(args.test_data_file)
    df = pd.read_json(args.test_data_file, lines=True) 
    df = df[['id', 'file_name', 'label']] 
    df["pred"] = y_preds
    df.to_json("results/test_preds.jsonlines", lines=True, orient='records', force_ascii=False)
        
             
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--feature_file_dir", default=None, type=str,
                        help="A data feature file input, containing the folder name of the sample feature folder.")
    parser.add_argument("--dsg_feature_file_dir", default=None, type=str,
                        help="A data feature file input, containing the folder name of the sample feature folder.")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_export", action='store_true',
                        help="Whether to save prediction output.")
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--CodeBERT_num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    parser.add_argument('--GAT_num_attention_heads', type=int, default=2,
                        help="number of attention heads used in GAT")
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                            help="Whether to write raw predictions on test data.")
    args = parser.parse_args()    
        
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    set_seed(args)  
    roberta_config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)   
    roberta_config.num_labels = 1
    roberta_config.num_attention_heads = args.CodeBERT_num_attention_heads
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name) 
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=roberta_config, ignore_mismatched_sizes=True)  
    GAT_out_channels=48 
    extended_model = ExtendedModel(
    model,
    roberta_config,  
    gat_heads=args.GAT_num_attention_heads, 
    gat_in_channels_cfg=768,  
    gat_in_channels_dfg=768,  
    gat_in_channels_dsg=768,
    gat_out_channels=GAT_out_channels, 
    hidden_dims=[GAT_out_channels*args.GAT_num_attention_heads*3+768, int((GAT_out_channels*args.GAT_num_attention_heads*3+768)/2), int((GAT_out_channels*args.GAT_num_attention_heads*3+768)/4)],  
    output_dim=2    
    )  
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = CodeGraphDataset(tokenizer, args, file_type='train')
        eval_dataset = CodeGraphDataset(tokenizer, args, file_type='eval')
        train(args, train_dataset, extended_model, tokenizer, eval_dataset)

    results = {}
    if args.do_eval:
        output_dir = os.path.join(args.output_dir,'checkpoint-best-f1', args.model_name)
        extended_model.load_state_dict(torch.load(output_dir),strict=False)
        extended_model.to(args.device)
        eval_dataset = CodeGraphDataset(tokenizer, args, file_type='eval')
        result=evaluate(args, extended_model, tokenizer, eval_dataset)   

    if args.do_test:
        if os.path.exists(args.model_name):
            output_dir = args.model_name
        else:
            checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
            if not os.path.exists(output_dir):
                output_dir = os.path.join(args.output_dir, args.model_name, 'checkpoint-best-f1', 'model.bin')
        extended_model.load_state_dict(torch.load(output_dir, map_location=args.device),strict=False)
        extended_model.to(args.device)
        test_dataset = CodeGraphDataset(tokenizer, args, file_type='test')
        test(args, extended_model, tokenizer, test_dataset, best_threshold=0.5)
                

if __name__ == "__main__":
    main()