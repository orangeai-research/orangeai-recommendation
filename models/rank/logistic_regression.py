# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved.
# The Logistic Regression implemention using torch 

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  

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
    


# test demo

# 假设的模型参数  
dense_feature_number = 4
sparse_feature_number = 3
embedding_dim = 2  
batch_size = 5  
  
# 初始化模型  
model = LogisticRegression(
    dense_feature_number, 
    sparse_feature_number, 
    embedding_dim)  
  
# 初始化dense_features (batch_size, dense_feature_number)  
# dense_features = None
dense_features = torch.randn(batch_size, dense_feature_number)  
  
# 初始化sparse_features (batch_size, sparse_feature_number)，并确保索引从1开始  
sparse_features = torch.randint(1, sparse_feature_number + 1, (batch_size, sparse_feature_number))  
  
# 进行前向运算  
output = model(dense_features, sparse_features)  
  
# 输出结果  
print(output)



def model_union_test():
    '''
        union test for deep model
    '''
    pass 
