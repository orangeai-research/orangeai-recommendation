# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved.

from __future__ import print_function
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class RecDataset(IterableDataset):
    '''
        criteo dataset
        agrs:
            file_dir = '/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_train_data_full'
            mode = "train" or "eval"
        
        thanks for paddlepaddle team suppy this 
    '''
    def __init__(self, file_list, mode="train"):
        super(RecDataset, self).__init__()
        self.file_list = [os.path.join(file_list, x) for x in os.listdir(file_list)]
        self.mode = mode
    
    def prepare_data(self):
        padding = 0
        sparse_slots = "click 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"
        self.sparse_slots = sparse_slots.strip().split(" ")
        self.dense_slots = ["dense_feature"]
        self.dense_slots_shape = [13]
        self.slots = self.sparse_slots + self.dense_slots
        self.slot2index = {}
        self.visit = {}
        for i in range(len(self.slots)):
            self.slot2index[self.slots[i]] = i
            self.visit[self.slots[i]] = False
        self.padding = padding
        
        self.data = []
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    line = l.strip().split(" ")
                    output = [(i, []) for i in self.slots]
                    for i in line:
                        slot_feasign = i.split(":")
                        slot = slot_feasign[0]
                        if slot not in self.slots:
                            continue
                        if slot in self.sparse_slots:
                            feasign = int(slot_feasign[1])
                        else:
                            feasign = float(slot_feasign[1])
                        output[self.slot2index[slot]][1].append(feasign)
                        self.visit[slot] = True
                    for i in self.visit:
                        slot = i
                        if not self.visit[slot]:
                            if i in self.dense_slots:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding] *
                                    self.dense_slots_shape[self.slot2index[i]])
                            else:
                                output[self.slot2index[i]][1].extend(
                                    [self.padding])
                        else:
                            self.visit[slot] = False
                    # sparse
                    output_list = []
                    for key, value in output[:-1]:
                        output_list.append(np.array(value).astype('int64'))
                    # dense
                    output_list.append(
                        np.array(output[-1][1]).astype("float32"))
                    # 
                    label = output_list[0]
                    sparse_feature = output_list[1:-1]
                    dense_feature = output_list[-1]

                    if self.mode == "train":
                        # print("label:", label)
                        # print("sparse_feature:", sparse_feature)
                        # print("dense_feature:", dense_feature)
                        yield (label, dense_feature, sparse_feature)
                    elif self.mode == "eval":
                        yield (dense_feature, sparse_feature)
                    else:
                        raise ValueError("mode shloud not be None!")

        
    def __iter__(self):
        return self.prepare_data()



# union test dataloader
# dataset = RecDataset(file_list='/Users/orange/code/orangeai-recommendation/dataset/criteo/slot_train_data_full')
# dataset = DataLoader(
#     dataset=dataset,
#     batch_size=2,
#     shuffle=False,
#     num_workers=0
# )

# for i, data in enumerate(dataset):
#     print("batch = ", i)
#     print("label:", data[0], "label shape:", data[0].shape, "\n")
#     print("sparse feature:", data[1], "sparse feature shape:", data[1][0].shape, "\n")
#     print("dense feature:", data[2], "dense feature shape", data[2].shape, "\n")

