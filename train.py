import os
import json
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN


class TextDataset(Dataset):
    """
    Emits batches of text
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.vocab_size = 50257
        C.block_size = 1024
        C.data_path = 'pile_data_10.jsonl'
        C.tokenizer_path = 'tokenizer_weights.pt'

        return C

    def __init__(self, config):
        self.config = config
        if os.path.exists(config.tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_path)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.save_pretrained(config.tokenizer_path)

        with open(config.data_path, 'r') as file:
            self.data = file.readlines()


    def get_vocab_size(self):
        return self.config.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = json.loads(self.data[idx]).get('text')   
        tokens = self.tokenizer(text)["input_ids"]

        # grab a chunk of (block_size + 1) characters from the data
        chunk = tokens[:self.config.block_size + 1]

        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pad the chunk if it is smaller than the block size
        if len(chunk) < self.config.block_size + 1:
            chunk += [self.tokenizer.eos_token_id] * (self.config.block_size + 1 - len(chunk))
        
        # return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train():
    print("Creating model")
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = 50257
    model_config.block_size = 1024
    model_config.checkpoint_path = None
    model = GPT(model_config)

    print("Loading dataset")
    data_config = TextDataset.get_default_config()
    train_dataset = TextDataset(data_config)

    print("Starting training")
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 1000
    train_config.batch_size = 16
    train_config.save_iterations = 10
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()

if __name__ == "__main__":
    train()
