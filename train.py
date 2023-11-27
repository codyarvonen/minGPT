import os
import json
import torch
import random

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
        C.tokenizer_path = 'tokenizer_weights_2.pt'
        C.perform_denoising = False

        return C

    def __init__(self, config):
        self.config = config
        if os.path.exists(config.tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(config.tokenizer_path)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.save_pretrained(config.tokenizer_path)

        # special_tokens_dict = {"regular": "[R]", "sequential": "[S]", "extreme": "[X]", "begin": "<B>", "space": "<S>", "end": "<E>"}
        # self.tokenizer.add_special_tokens(special_tokens_dict)

        self.tokenizer.add_tokens(["[R]", "[S]", "[X]", "<B>", "<S>", "<E>"], special_tokens=True)
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        # model.resize_token_embeddings(len(tokenizer))

        with open(config.data_path, 'r') as file:
            self.data = file.readlines()


    def get_vocab_size(self):
        return self.config.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data)
    
    def regular_denoising(self, tokens):
        # Implement regular span corruption task
        # Sample spans with a mean length of 3 and corruption rate of 15%
        tokens.insert(0, self.tokenizer.convert_tokens_to_ids("[R]"))
        if len(tokens) > self.config.block_size:
            tokens = tokens[:self.config.block_size]

        span_length = max(2, int(random.gauss(3, 1)))
        corruption_rate = 0.15
        num_corrupt_seqs = int(corruption_rate * len(tokens)) / span_length
        corrupt_indices = random.sample(range(len(tokens) - span_length), num_corrupt_seqs)
        for idx in corrupt_indices:
            tokens[idx] = [self.tokenizer.mask_token]
            tokens[idx + 1:idx + span_length] = [-1] * (span_length - 1)
        tokens = tokens[tokens != -1]
        targets = [self.tokenizer.convert_tokens_to_ids("<B>")]
        for i in range(num_corrupt_seqs):
            targets.append(self.tokenizer.mask_token)
            targets.append(self.tokenizer.convert_tokens_to_ids("<S>"))
        targets.append(self.tokenizer.convert_tokens_to_ids("<E>"))
        return tokens, targets

    def extreme_denoising(self, tokens):
        # Implement extreme denoising task
        # Sample spans with a mean length of 32 or corruption rate of up to 50%
        tokens.insert(0, self.tokenizer.convert_tokens_to_ids("[X]"))
        if len(tokens) > self.config.block_size:
            tokens = tokens[:self.config.block_size]
        corruption_rate = 0.5
        if random.random() < 0.5:
            span_length = max(2, int(random.gauss(32, 1)))
            corruption_rate = 0.15
        else:
            span_length = max(2, int(random.gauss(3, 1)))
            corruption_rate = min(0.5, random.uniform(0, 0.5))

        num_corrupt_seqs = int(corruption_rate * len(tokens)) / span_length
        corrupt_indices = random.sample(range(len(tokens) - span_length), num_corrupt_seqs)
        for idx in corrupt_indices:
            tokens[idx] = [self.tokenizer.mask_token]
            tokens[idx + 1:idx + span_length] = [-1] * (span_length - 1)
        tokens = tokens[tokens != -1]
        targets = [self.tokenizer.convert_tokens_to_ids("<B>")]
        for i in range(num_corrupt_seqs):
            targets.append(self.tokenizer.mask_token)
            targets.append(self.tokenizer.convert_tokens_to_ids("<S>"))
        targets.append(self.tokenizer.convert_tokens_to_ids("<E>"))
        return tokens, targets

    def sequential_denoising(self, tokens):
        # Implement sequential denoising (PrefixLM) task
        # Noise is sampled from the start of the text to a randomly sampled point
        tokens.insert(0, self.tokenizer.convert_tokens_to_ids("[S]"))
        if len(tokens) > self.config.block_size:
            tokens = tokens[:self.config.block_size]
        span_length = random.randint(0, len(tokens) - 1)
        tokens[span_length] = self.tokenizer.mask_token
        tokens[span_length + 1:] = [-1] * (span_length - 1)
        tokens = tokens[tokens != -1]
        targets = [self.tokenizer.convert_tokens_to_ids("<B>"), self.tokenizer.mask_token, self.tokenizer.convert_tokens_to_ids("<E>")]
        return tokens, targets

    def __getitem__(self, idx):
        text = json.loads(self.data[idx]).get('text')   
        tokens = self.tokenizer(text)["input_ids"]

        if not self.config.perform_denoising:
            # grab a chunk of (block_size + 1) characters from the data
            chunk = tokens[:self.config.block_size + 1]

            # Pad the chunk if it is smaller than the block size
            if len(chunk) < self.config.block_size + 1:
                chunk += [self.tokenizer.eos_token_id] * (self.config.block_size + 1 - len(chunk))

            # return as tensors
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            return x, y
        else:

            chunk = tokens[:self.config.block_size]

            # Pad the chunk if it is smaller than the block size
            if len(chunk) < self.config.block_size:
                chunk += [self.tokenizer.eos_token_id] * (self.config.block_size - len(chunk))

            masked_tokens, target = []
            # Apply denoising tasks
            if random.random() < 0.5:
                masked_tokens, target = self.sequential_denoising(chunk)
            elif random.random() < 0.75:
                masked_tokens, target = self.extreme_denoising(chunk)
            else:
                masked_tokens, target = self.regular_denoising(chunk)
            
            # return as tensors
            x = torch.tensor(masked_tokens, dtype=torch.long)
            y = torch.tensor(target, dtype=torch.long)
            return x, y


def train():
    print("Creating model")
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = 50263
    model_config.block_size = 1024
    model_config.checkpoint_path = 'checkpoint_1999.pt'
    model = GPT(model_config)

    print("Loading dataset")
    data_config = TextDataset.get_default_config()
    train_dataset = TextDataset(data_config)

    print("Starting training")
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4
    train_config.max_iters = 4000
    train_config.batch_size = 16
    train_config.save_iterations = 100
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()

if __name__ == "__main__":
    train()
