# coding=utf-8
from __future__ import unicode_literals

import gc

import torch

from Model.Config import Config
import os.path
from Model.MyModel import MyModel
from utils.process_data import load_data

DATA_DIR = 'data_with_emotion'
SAVE_DIR = 'saved'
datasets = [
    {"name":'r-19',
     'language':'english',
     'embedding_path':'resource/embedding/glove.6B.100d.txt',
     'embedding_dim': 100,
     'class_num':3,
     'batch_size':4,
     'epochs':20
    },
    {"name":'weibo',
     'language':'chinese',
     'embedding_path':'resource/embedding/sgns.weibo.word/sgns.weibo.word',
     'embedding_dim': 300,
     'batch_size':8,
     'class_num':2,
     'epochs':20
    }
]





if __name__ == '__main__':

    for dataset in datasets:
        name = dataset['name']
        print('#'*60)
        print('#'*60)
        print(f'数据集 {name} 开始训练')
        config = Config(dataset['class_num'], dataset['embedding_dim'])

        dataset_dir = os.path.join(DATA_DIR, dataset['name'])
        train_data, val_data, test_data = load_data(dataset_dir)
        SAVED_MODEL_DIR = os.path.join(SAVE_DIR,dataset['name'])
        if not os.path.exists(SAVED_MODEL_DIR):
            os.mkdir(SAVED_MODEL_DIR)
        model = MyModel(config)
        model.train(train_data=train_data,
                    val_data=val_data,
                    batch_size=dataset['batch_size'],
                    lr=1e-3,
                    epochs=dataset['epochs'],
                    embedding_path=dataset['embedding_path'],
                    saved_dir=SAVED_MODEL_DIR
                    )
        print(f'数据集 {name} 训练结束')
        print("\n")
        # 训练完第一个数据集后
        del model  # 删除模型
        del train_data, val_data, test_data  # 删除数据集
        torch.cuda.empty_cache()  # 清空 GPU 缓存（如果使用 GPU）
        gc.collect()  # 触发垃圾回收

