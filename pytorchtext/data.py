import os
import string
from typing import List

import numpy as np
import requests
import torch
from functional import seq
from torch.utils.data import Dataset


def main():
    for size in [0, 1, 2, 5, 100]:
        fib_seq = get_fibonacci_seq(size=size)
        print(len(fib_seq), fib_seq)


class DatasetText(Dataset):
    def __init__(
        self,
        token_id_list: List[int],
        batch_size: int,
        seq_len: int,
    ) -> None:
        self.token_tensor = torch.tensor(token_id_list, dtype=torch.long)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        full_seq_count = (len(token_id_list) - 1) // seq_len

        self.x = self.token_tensor[: full_seq_count * seq_len].reshape(-1, seq_len)
        self.y = self.token_tensor[1 : full_seq_count * seq_len + 1].view(-1, seq_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_batch(self):
        ix = torch.randint(0, len(self), (self.batch_size,))
        x = torch.stack([self.token_tensor[i : i + self.seq_len] for i in ix]).to(
            self.device
        )
        y = torch.stack(
            [self.token_tensor[i + 1 : i + self.seq_len + 1] for i in ix]
        ).to(self.device)
        return x, y


class Tokenizer:
    def __init__(self) -> None:
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.vocab = []

    def fit(self, text: str):
        text_seq = seq(self.get_token_list(text=text))

        self.token_to_id = text_seq.distinct().zip_with_index().to_dict()
        self.id_to_token = (
            seq(self.token_to_id.items()).map(lambda kv: (kv[1], kv[0])).to_dict()
        )

        self.vocab = self.token_to_id.keys()
        self.vocab_size = len(self.token_to_id)

    def tokenize(self, text: str) -> List[int]:
        return (
            seq(self.get_token_list(text))
            .map(lambda token: self.token_to_id[token])
            .to_list()
        )

    def tokenize_back(self, id_list: List[int]) -> str:
        return "".join(
            (seq(id_list).map(lambda token: self.id_to_token[token]).to_list())
        )

    @staticmethod
    def get_token_list(text: str) -> List[str]:
        return list(text.lower())


def build_data_random(text_size: int, batch_size: int, seq_len: int):
    np.random.seed(0)
    text = "".join(
        np.random.choice(list(string.ascii_letters + " "), size=text_size, replace=True)
    )
    tokenizer = Tokenizer()
    tokenizer.fit(text)

    dataset = DatasetText(
        token_id_list=tokenizer.tokenize(text), batch_size=batch_size, seq_len=seq_len
    )

    return dataset, tokenizer


def get_shakespeare(batch_size: int, seq_len: int):
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r") as f:
        data = f.read()
    n = len(data)
    # train_data = data[:int(n * 0.9)]
    # val_data = data[int(n * 0.9):]
    tokenizer = Tokenizer()
    tokenizer.fit(data)

    dataset = DatasetText(
        token_id_list=tokenizer.tokenize(data), batch_size=batch_size, seq_len=seq_len
    )

    return dataset, tokenizer


def get_fibonacci_seq(size: int, seq: List[int] = [1]):
    if size == 0:
        return []
    if size == 1:
        return seq
    next_element = sum(seq[-2:])
    return get_fibonacci_seq(size=size - 1, seq=seq + [next_element])


if __name__ == "__main__":
    main()
