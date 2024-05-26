# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved.
# The train and eval for OrangeRec
# author: orange
# create time: 2024-05-25


import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
from torch.utils.tensorboard import SummaryWriter  
from torch.utils.data import DataLoader
import numpy as np
import os 
import logging
from models.rank.wide_deep import WideDeep
from dataset.criteo.criteo_dataloader import RecDataset
from datetime import datetime
from tqdm import tqdm 


class Custom_LRScheduler(_LRScheduler):
    """
        custom learning rate scheduler
    """
    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = ..., verbose: bool = ...) -> None:
        super().__init__(optimizer, last_epoch, verbose)
        pass 


# logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  
logger = logging.getLogger(__name__)  

# hyper params

train_data_dir  = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_train_data_full'
eval_data_dir = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_test_data_full'
model_save_checkpoint_dir='/Users/orange/code/orangeai-recommendation/output'
batch_size = 1024
learning_rate = 0.001
epoches = 5
interval_batch = 5
lr_scheduler_epoch_size = 1
device="cuda"

if torch.cuda.is_available():  
    print("CUDA is available! Training on GPU ...")  
    device = torch.device("cuda")  
else:  
    print("CUDA is not available. Training on CPU ...")  
    device = torch.device("cpu")

train_data_loader = DataLoader(
    RecDataset(file_list = train_data_dir, mode="train"), 
    batch_size=batch_size, 
    shuffle=False)  

val_data_loader = DataLoader(
    RecDataset(file_list = eval_data_dir, mode="eval"), 
    batch_size=batch_size, 
    shuffle=False)  


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
  
# 定义优化器和损失函数  
"""
RuntimeError: SparseAdam does not support dense gradients, please consider Adam instead
"""

optimizer = optim.Adam(model.parameters(), lr=learning_rate)  
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_epoch_size, gamma=0.1)
#optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
'''
raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
RuntimeError: Adam does not support sparse gradients, please consider SparseAdam instead
'''
# criterion = nn.BCELoss()  # 例如，对于二分类问题使用二元交叉熵损失  
criterion = nn.BCEWithLogitsLoss()  # 例如，对于二分类问题使用二元交叉熵损失  






def train_eval_loop(model, 
                    train_data_dir, 
                    eval_data_dir, 
                    batch_size, 
                    optimizer, 
                    scheduler, 
                    criterion, 
                    epoches, 
                    writer, 
                    interval_batch, 
                    model_save_checkpoint_dir,
                    device
                    ):
    """
        model train and eval and save
    """
    train_iter_batch_num =  0
    eval_iter_batch_num = 0
    EVAL_BEST_AUC = 0.0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(epoches):
        logger.info(f'Starting epoch {epoch + 1}/{epoches}')  
        train_data_loader = DataLoader(
        RecDataset(file_list = train_data_dir, mode="train"), 
        batch_size=batch_size, 
        shuffle=False)  
        eval_data_loader = DataLoader(
            RecDataset(file_list = eval_data_dir, mode="eval"), 
            batch_size=batch_size, 
            shuffle=False)  
        
        model = model.to(device)
        model.train()
        train_mini_batch_loss = 0.0
        train_loss = 0.0
        train_mini_batch_y_true = [] # mini batch
        train_mini_batch_y_pred = [] # mini batch
        for batch_idx, data in enumerate(tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{epoches}")):
            if epoch == 0:
                train_iter_batch_num = batch_idx  + 1 #record the one whole epoch train dataloader batch nums
            data = data.to(device)
            label, dense_feature, sparse_feature= data[0], data[1], data[2]
            optimizer.zero_grad()
            logits = model(dense_feature, sparse_feature)
            train_mini_batch_y_true.extend(label.detach().numpy())  
            train_mini_batch_y_pred.extend(torch.sigmoid(logits).detach().numpy())  
            loss = criterion(logits, label.float())
            loss.backward()
            optimizer.step()
            #record the loss
            train_mini_batch_loss += loss.item() # record mini batch train loss
            train_loss += loss.item() # record total one epoch train loss
            if batch_idx % interval_batch == 0:
                train_batch_avg_loss = train_mini_batch_loss / interval_batch # average of train batch loss 
                #print('epoch [{}/{}] batch {} train mini batch avg loss: {}'.format(epoch + 1, epoches, batch_idx + 1, train_batch_avg_loss)) 
                auc, accuracy, precision, recall, f1 = caculate_metrics(train_mini_batch_y_true, train_mini_batch_y_pred)
                logger.info(f'Epoch [{epoch + 1}/{epoches}] Batch {batch_idx + 1} train mini batch avg loss: {train_batch_avg_loss} AUC:{auc} ACC:{accuracy} Precision:{precision} F1-Score:{f1}') 
                tb_x = epoch * train_iter_batch_num + batch_idx + 1
                writer.add_scalar('Loss/Train', train_batch_avg_loss, tb_x)
                writer.add_scalar('AUC/Train', auc, tb_x)  
                writer.add_scalar('Accuracy/Train', accuracy, tb_x)  
                writer.add_scalar('Precision/Train', precision, tb_x)  
                writer.add_scalar('Recall/Train', recall, tb_x)  
                writer.add_scalar('F1 Score/Train', f1, tb_x)  
                train_mini_batch_loss = 0.0
        #print('epoch [{}/{}] train one epoch loss: {}'.format(epoch + 1, epoches, train_loss / train_iter_batch_num))
        # update the learning rate of current optimizer
        scheduler.step()
        # get the learning rate of current optimizer
        for param_group in optimizer.param_groups:  
            lr = param_group['lr']  
        writer.add_scalar('Train Learning Rate', lr, epoch)
        logger.info(f'Epoch [{epoch + 1}/{epoches}] train loss: {train_loss / train_iter_batch_num} train learning rate: {lr}')  

        model.eval()
        eval_mini_batch_loss = 0.0
        eval_batch_y_true = [] # mini batch
        eval_batch_y_pred = [] # mini batch
        eval_loss = 0.0 # epoch
        y_true = []  # epoch 
        y_pred = []  # epoch 
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(eval_data_loader, desc=f"Epoch {epoch+1}/{epoches}")):
                if epoch == 0:
                    eval_iter_batch_num = batch_idx + 1 #record the one whole epoch eval dataloader batch nums
                data = data.to(device)
                label, dense_feature, sparse_feature= data[0], data[1], data[2]
                logits = model(dense_feature, sparse_feature)
                loss = criterion(logits, label.float())
                eval_mini_batch_loss += loss.item()
                eval_loss += loss.item()
                y_true.extend(label.detach().numpy())  
                y_pred.extend(torch.sigmoid(logits).detach().numpy().flatten())  
                # if batch_idx % interval_batch == 999:
                #     #batch loss
                #     eval_batch_avg_loss = eval_mini_batch_loss / interval_batch # average of train batch loss 
                #     print('epoch {} mini batch {} eval loss: {}'.format(epoch + 1, batch_idx + 1, eval_batch_avg_loss)) 
                #     tb_x = epoch * eval_iter_batch_num+ batch_idx + 1
                #     tb_writer.add_scalar('Loss/eval', eval_batch_avg_loss, tb_x)
                #     #eval_batch_loss = 0.0
                #     #batch metrics

            # epoch eval: 
            auc, accuracy, precision, recall, f1 = caculate_metrics(y_true, y_pred)
            # record every epoch train and eval TensorBoard logging 
            writer.add_scalar('Loss/eval', eval_loss / eval_iter_batch_num, epoch)  
            writer.add_scalar('AUC/eval', auc, epoch)  
            writer.add_scalar('Accuracy/eval', accuracy, epoch)  
            writer.add_scalar('Precision/eval', precision, epoch)  
            writer.add_scalar('Recall/eval', recall, epoch)  
            writer.add_scalar('F1 Score/eval', f1, epoch)  

            logger.info(f'Epoch {epoch+1}/{epoches}, Eval Loss: {eval_loss / eval_iter_batch_num:.4f}, AUC: {auc:.4f}, '  
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')  
            #save the model with best auc and best performance
            if auc > EVAL_BEST_AUC:
                EVAL_BEST_AUC = auc
                if model_save_checkpoint_dir:
                    model_path = os.path.join(model_save_checkpoint_dir, 'model_{}_{}'.format(timestamp, epoch))
                    torch.save(model.state_dict(), model_path)
                    #print("model save successfully!!")
                    logger.info('Model performance improved. Saving checkpoint...') 
                else:
                    raise ValueError("model checkpoint path is null, please check your checkpoint path!!")
        

def caculate_metrics(y_true:list, y_pred:list)->list:
    """
        caculate the metircs: AUC, ACC, Precision, Recall, F1-Score, etc...
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score  
    auc = roc_auc_score(y_true, y_pred) 
    #print("y_true:", y_true, "y_pred:", y_pred, "auc:", auc)
    y_pred = np.array(y_pred) 
    accuracy = accuracy_score(y_true, (y_pred > 0.5).astype(int))  
    precision = precision_score(y_true, (y_pred > 0.5).astype(int))  
    recall = recall_score(y_true, (y_pred > 0.5).astype(int))  
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))  
    metrics_list = [float(auc), float(accuracy), float(precision), float(recall), float(f1)]
    return metrics_list


# def record_tensorborad():
#         writer.add_scalar('Loss/eval', eval_loss / eval_iter_batch_num, epoch)  
#         writer.add_scalar('AUC/eval', auc, epoch)  
#         writer.add_scalar('Accuracy/eval', accuracy, epoch)  
#         writer.add_scalar('Precision/eval', precision, epoch)  
#         writer.add_scalar('Recall/eval', recall, epoch)  
#         writer.add_scalar('F1 Score/eval', f1, epoch)  


# TensorBoard setting 
writer = SummaryWriter(log_dir='runs/my_experiment')
train_eval_loop(
    model=model,
    train_data_dir=train_data_dir, 
    eval_data_dir=eval_data_dir,
    batch_size=batch_size,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    epoches=epoches,
    writer=writer,
    interval_batch=interval_batch,
    model_save_checkpoint_dir=model_save_checkpoint_dir,
    device=device
)
writer.close()