# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved. 
# author: orange
# create time: 2024-05-05

import torch
import torch.nn as nn
import math

class WideDeep(nn.Module):
    """
        wide&deep pytorch version
        paper ref:  https://dl.acm.org/doi/pdf/10.1145/2988450.2988454 [DLRS 2016] Wide & Deep Learning for Recommender Systems 
    """
    def __init__(self, 
                 dense_feature_num: int = 13, 
                 sparse_feature_num: int = 26,
                 sparse_feature_embedding_num: int = 10241024, 
                 sparse_feature_embedding_dim: int = 9, 
                 use_sparse: bool = True,
                 hidden_units: list = [512, 256, 128, 32],
                 activate_fun: str = 'relu'
                 ):
        super(WideDeep, self).__init__()
        self.dense_feature_num = dense_feature_num
        self.sparse_feature_num = sparse_feature_num
        self.sparse_feature_embedding_num = sparse_feature_embedding_num
        self.sparse_feature_embedding_dim = sparse_feature_embedding_dim
        self.use_sparse = use_sparse
        self.hidden_units = hidden_units
        self.activate_fun =  activate_fun
        # wide layer forward 
        self.wide_layer = nn.Linear(
            in_features=self.dense_feature_num,
            out_features=1
        )
        # init sparse features embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.sparse_feature_embedding_num,
            embedding_dim=self.sparse_feature_embedding_dim,
            sparse=self.use_sparse
        )
        # linear layers: [26 * 9 + 13, 512, 256, 128, 32, 1]
        hidden_units_list =  [self.sparse_feature_num * self.sparse_feature_embedding_dim + self.dense_feature_num] + self.hidden_units + [1]
        # activate layers: use relu activate function, actually  we just need 4 relu layers between 5 hidden fc layers 
        # activate_fun_list = [nn.ReLU() for _ in range(len(self.hidden_units)) if self.activate_fun=='relu']
        # init deep hidden layer
        hidden_layers = []
        for i in range(len(hidden_units_list) - 1):
            in_features = hidden_units_list[i]
            out_features = hidden_units_list[i  + 1]
            # create linear layer
            linear_layer = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
            # init linear layer
            nn.init.normal_(linear_layer.weight, mean=0, std=1.0 / math.sqrt(in_features))
            hidden_layers.append(linear_layer)
            # # add activate function, and we don`t need add the act layer in last hidden linear layer
            # if activate_fun_list is not None:
            #     hidden_layers.append(activate_fun_list.pop(0))
            # add activate function, but not in the last hidden layer  
            if i < len(hidden_units_list) - 2 and activate_fun == 'relu':  
                hidden_layers.append(nn.ReLU())  
        # deep layer
        self.deep_layer = nn.Sequential(*hidden_layers)
        
    def forward(self, dense_feature: torch.Tensor, sparse_feature: list) -> torch.Tensor:
        # in wide&deep forward stage, firstlly, dense feature sholud pass the wide layer and map into one nerual unit.
        wide_layer_output = self.wide_layer(dense_feature)
        # in sparse forward part, every sparse feature colmun need pass the embedding layer which will be mapped into dense embedding vector.
        sparse_tensor_embedding_list = []
        for sparse_tensor in sparse_feature:
            sparse_tensor_embedding = self.embedding(sparse_tensor.long()) # (N, 1) -> (N, 1, 9) , N is the batch size and 9 is the sparse_feature_embedding_dim
            sparse_tensor_embedding = sparse_tensor_embedding.view(-1, self.sparse_feature_embedding_dim) # to keep standard input shape we could change the tensor shape from (N, 1, 9) -> (N, 1*9)
            sparse_tensor_embedding_list.append(sparse_tensor_embedding)
        # Combine sparse embeddings and dense feature
        # sparse_tensor_embedding_list shape now is (N, 26 * 9), and we need to append the dense feature to combine a big one.
        # print("sparse_tensor_embedding_list:", sparse_tensor_embedding_list)
        # print("dense_feature:", dense_feature)
        combined_features = torch.cat(tensors=sparse_tensor_embedding_list + [dense_feature], dim=1)
        # deep layer forward  
        deep_layer_output = combined_features
        for mlp_layer in self.deep_layer:
            deep_layer_output = mlp_layer(deep_layer_output)
        # Combine wide and deep outputs 
        logits = torch.add(wide_layer_output, deep_layer_output) # (N, 1)
        return logits




        


