# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved.
# The Logistic Regression implemention using torch 

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math

class LogisticRegression(nn.Module):  
    '''
        args:
            dense_feature_number: number of the dense features, 
            sparse_feature_number: number of the sparse features, 
            embedding_dim: embedding dim
        supply:
            if you forget the details about nn.Embedding() , 
            you can visit https://blog.csdn.net/lsb2002/article/details/132993128.

    '''
    def __init__(self, dense_feature_number=0, sparse_feature_number=0, embedding_dim=0):  
        super(LogisticRegression, self).__init__()  
         # init some commen parameters 
        self.dense_feature_number = dense_feature_number  
        self.sparse_feature_number = sparse_feature_number  
        if sparse_feature_number > 0 and embedding_dim > 0:  
            self.embedding_dim = embedding_dim 
            self.embedding_layer = nn.Embedding(  
                num_embeddings= 1 + sparse_feature_number,  # avoid the index out of range
                embedding_dim=embedding_dim,  
                padding_idx=0  
            )  
        # init the logistic rregression parameters
        self.linear_layer = nn.Linear(self.dense_feature_number + self.sparse_feature_number * embedding_dim if sparse_feature_number > 0 else self.dense_feature_number, 1)  
        self.bias = nn.Parameter(torch.zeros(1))  
  
    def forward(self, dense_features: torch.Tensor = None, sparse_features: torch.Tensor = None) -> torch.Tensor:  
        if dense_features is None and sparse_features is None:  
            raise ValueError("Input features should not be None!")  
        feature = None  
        if dense_features is not None and sparse_features is not None:  
            sparse_embeddings = self.embedding_layer(sparse_features)  
            sparse_embeddings = sparse_embeddings.view(sparse_features.shape[0], -1)  # 可能需要重塑，具体取决于sparse_features的形状  
            feature = torch.cat([dense_features, sparse_embeddings], dim=1)  
        elif dense_features is not None:  
            feature = dense_features  
        elif sparse_features is not None:  
            sparse_embeddings = self.embedding_layer(sparse_features)  
            sparse_embeddings = sparse_embeddings.view(sparse_features.shape[0], -1)
            feature = sparse_embeddings  
        
        # start the logistic regression
        output = self.linear_layer(feature) 
        output = output + self.bias
        output = torch.sigmoid(output) # the formula of the logistic regression
        return output.squeeze()
    

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
        print("sparse_tensor_embedding_list:", sparse_tensor_embedding_list)
        print("dense_feature:", dense_feature)
        combined_features = torch.cat(tensors=sparse_tensor_embedding_list + [dense_feature], dim=1)
        # deep layer forward  
        deep_layer_output = combined_features
        for mlp_layer in self.deep_layer:
            deep_layer_output = mlp_layer(deep_layer_output)
        # Combine wide and deep outputs 
        logits = torch.add(wide_layer_output, deep_layer_output) # (N, 1)
        return logits
