from Model.Config import Config
import os.path
from Model.MyModel import MyModel
from utils.process_data import load_data

MODEL_DIR = 'saved'

DATA_DIR = 'data_with_emotion'
SAVE_DIR = 'saved'
datasets = [
    {"name":'weibo',
     'language':'chinese',
     'embedding_path':'resource/embedding/sgns.weibo.word/sgns.weibo.word',
     'embedding_dim': 300,
     'class_num':2
    },
    {"name":'r-19',
     'language':'english',
     'embedding_path':'resource/embedding/glove.6B.100d.txt',
     'embedding_dim': 100,
     'class_num':3
    }
]
if __name__ == '__main__':

    for dataset in datasets:
        name = dataset['name']
        print('#'*60)
        print('#'*60)
        print(f'数据集 {name} 开始测试')
        config = Config(dataset['class_num'], dataset['embedding_dim'])

        dataset_dir = os.path.join(DATA_DIR, dataset['name'])
        model_dir = os.path.join(MODEL_DIR, dataset['name'])

        train_data, val_data, test_data = load_data(dataset_dir)
        model = MyModel(config)
        model.load_model(model_path=os.path.join(model_dir,'best_model.pth'))
        model.test(test_data)