import os
import json
import torch
import random
from datetime import datetime

from torch.utils.data import Dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN


def generate():
    print("Creating model")
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.block_size = 10000
    model_config.num_features = 6
    model_config.checkpoint_path = "checkpoint_999.pt"
    model = GPT(model_config)

    model.eval()
    timestamp = datetime.now().timestamp() - datetime.fromisoformat("2020-01-01T00:00:00+00:00").timestamp()
    result = model.generate(torch.tensor([[timestamp, 0.0, 1375.0, 85, 1, 21.2]], dtype=torch.float), 10000)

    print(result.tolist())



if __name__ == "__main__":
    generate()
