import urllib.request
import os
import re
from importlib.metadata import version
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch


print("tiktoken version:", version("tiktoken"))

file_path = "the-verdict.txt"


if not os.path.isfile(file_path):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# max_length = context
# stride = no of positions shifted to the right
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.inputs = []
        self.targets = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            self.inputs.append(torch.tensor(token_ids[i:i+max_length]))
            self.targets.append(torch.tensor(token_ids[i+1:i+max_length+1]))

    def __len__(self,):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
            )

    return dataloader


dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)
# second_batch = next(data_iter)
# print(second_batch)

