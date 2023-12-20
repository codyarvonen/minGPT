import os
import json
import torch
import random
from datetime import datetime

from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN


class ActivityDataset(Dataset):
    """
    Emits batches of text
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 10000
        C.data_path = '../data'

        return C

    def __init__(self, config):
        self.config = config
        self.base_time = datetime.fromisoformat("2020-01-01T00:00:00+00:00").timestamp()
        self.data = []

        for filename in os.listdir(config.data_path):
            dir = os.path.join(config.data_path, filename)
            if os.path.isfile(dir):
                with open(dir, 'r') as file:
                    self.data.append(json.load(file))

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_obj = self.data[idx]

        timestamp = datetime.fromisoformat(data_obj["start_date"]).timestamp() - self.base_time
        workout_type = 0 if data_obj["workout_type"] is None else data_obj["workout_type"]
        total_dist = data_obj["total_distance"]
        time = data_obj["time"]
        dist = data_obj["distance"]
        elev = data_obj["altitude"] if "altitude" in data_obj else [0] * len(time)
        hr = data_obj["heartrate"] if "heartrate" in data_obj else [0] * len(time)

        x = [[timestamp + time[i], dist[i], elev[i], hr[i], workout_type, total_dist] for i in range(len(time))]
        x += [[0, 0, 0, 0, 0, 0]] * (self.config.block_size - len(x))
        y = x[1:] + ([[0, 0, 0, 0, 0, 0]] * (self.config.block_size - len(x) + 1))

        # return as tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
        

def train():
    print("Creating model")
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.block_size = 10000
    model_config.num_features = 6
    model_config.checkpoint_path = None
    model = GPT(model_config)

    print("Loading dataset")
    data_config = ActivityDataset.get_default_config()
    train_dataset = ActivityDataset(data_config)
    # for i in range(len(train_dataset)):
    #     if train_dataset[i][0].tolist()[0][5] > 40000:
    #         print(train_dataset[i])
    #         print(len(train_dataset[i][0].tolist()))
    #         print(len(train_dataset[i][1].tolist()))

    print("Starting training")
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 10000
    train_config.batch_size = 16
    train_config.save_iterations = 1000
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()

if __name__ == "__main__":
    train()
