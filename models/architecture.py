# Copyright (c) 2024 OrangeRec Authors. All Rights Reserved.
# This is the design of the model architecture 
# We use the ABC (Abstract Base Classes, ABCs) define the base class of the model
# to adapt to the different tasks while using the same interface.

from abc import ABC, abstractmethod

class OrangeRecAbstractBaseClass(ABC):
    '''
        The abstract base class 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def task(self):
        '''
            we have rank tasks and recall tasks and multitask tasks
        '''
        pass 


class OrangeRecRank(OrangeRecAbstractBaseClass):
    '''
        The rank task
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def task(self):

        return super().task()


class OrangeRecRecall(OrangeRecAbstractBaseClass):
    '''
        The recall task
    '''

    def __init__(self) -> None:
        super().__init__()

    def task(self):
        return super().task()
    
class OrangeRecMultiTask(OrangeRecAbstractBaseClass):
    '''
        The multi-task 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def task(self):
        return super().task()
    


class ModelArgs(object):
    '''
        model arguments 
    '''
    def __init__(self, dense_feature_number=0, sparse_feature_number=0, embedding_dim=0) -> None:

        self.dense_feature_number = dense_feature_number  
        self.sparse_feature_number = sparse_feature_number  
