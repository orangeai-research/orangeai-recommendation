# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved. 
# author: orange
# create time: 2024-05-22
# desc: the core trainer of OrangeRec

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from dataset.criteo.criteo_dataloader import RecDataset
from models.rank.wide_deep import WideDeep
import torch.optim as optim
import torch.nn as nn 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score  

class AbstractTrainer(ABC):
    """
       The Abstract Class for OrangeRec trainer. 
       In this class, we define the standard api for model training and evaluating and
       optimizer and loss function configuration.
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train_dataloader_setting(self):
        """
            return a train dataloader 
        """
        pass 

    @abstractmethod
    def eval_dataloader_setting(self):
        """
            return a eval dataloader 
        """
        pass 
    
    @abstractmethod
    def model_setting(self):
        """
        deep learning model configuration
        """
        pass 
    
    @abstractmethod
    def optimizer_setting(self):
        """
        optimizer configuration
        """
        pass 

    @abstractmethod
    def loss_function_setting(self):
        """
        loss function configuration
        """
        pass 
   
    @abstractmethod
    def metrics_setting(self):
        """
        model performance measurement & metrics configuration
        """
        pass 

    @abstractmethod
    def train_eval_loop(self):
        """
            train and eval loop
        """
        pass

    
class HyperParameters(object):
    """
        hyper parameters for training
    """
    def __init__(self,
                train_data_dir,
                eval_data_dir,
                batch_size,
                shuffle,
                epoches,
                learning_rate
                ):
        self.train_data_dir = train_data_dir
        self.eval_data_dir = eval_data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoches = epoches
        self.learning_rate = learning_rate
        


class BasicTrainer(AbstractTrainer):
    """
        a basic trainer
    """
    def __init__(self, hyper_params: HyperParameters, model,  ) :
        super().__init__()
        self.train_data_dir = hyper_params.train_data_dir
        self.eval_data_dir = hyper_params.eval_data_dir
        self.batch_size = hyper_params.batch_size
        self.shuffle = hyper_params.shuffle
        self.epoches = hyper_params.epoches
        self.learning_rate = hyper_params.learning_rate

        self.model = model

    def train_dataloader_setting(self):
        # train_data_dir  = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_train_data_full'
        # eval_data_dir = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_test_data_full'
        train_data_loader = DataLoader(
            RecDataset(file_list=self.train_data_dir, mode="train"), 
            batch_size=self.batch_size, 
            shuffle=False)  
        return train_data_loader
    
    def eval_dataloader_setting(self):
        # train_data_dir  = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_train_data_full'
        # eval_data_dir = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_test_data_full'
        eval_data_loader = DataLoader(
            RecDataset(file_list=self.eval_data_dir, mode="eval"), 
            batch_size=self.batch_size, 
            shuffle=False)  
        return eval_data_loader
    
    def model_setting(self):
        # 实例化模型  
        model = WideDeep(
            dense_feature_num = 13,
            sparse_feature_num = 26,
            sparse_feature_embedding_num = 10241024,
            sparse_feature_embedding_dim  = 9,
            use_sparse = False,
            hidden_units = [512, 256, 128, 32],
            activate_fun = 'relu'
        )  
        return model
    

    def optimizer_setting(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  
        return optimizer
    
    def loss_function_setting(self):
        loss = nn.BCEWithLogitsLoss() 
        return loss
    

    def train_eval_loop(self):
        
        for epoch in range(self.epoches):
            self.model.train()
            train_batch_loss = 0.0

            


    









    
    