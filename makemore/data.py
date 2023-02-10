from collections import Counter
from copy import copy
from typing import List, Tuple

import torch
from functional import seq
from matplotlib import pyplot as plt
from torch import Tensor

SEP = "."


def get_bigram_list(word_list: List[str]) -> list:
    def from_word(word: str) -> list:
        word = SEP + word + SEP
        return [(char_1, char_2) for char_1, char_2 in zip(word, word[1:])]

    return seq(word_list).flat_map(from_word).list()


def get_count(bigram_list):
    return Counter(bigram_list)


def get_dict_id_char(word_list) -> Tuple[dict, dict]:
    char_list = [SEP] + sorted(list(set("".join(word_list))))

    def reverse_tuple(t):
        return t[::-1]

    id_to_char = dict(enumerate(char_list))
    char_to_id = dict(seq(enumerate(char_list)).map(reverse_tuple))
    return id_to_char, char_to_id


def get_matrix_count(bigram_count_list: list, char_to_id: dict) -> Tensor:
    n_char = len(char_to_id)
    N = torch.zeros(size=(n_char, n_char), dtype=torch.int32)
    for (char_1, char_2), n in bigram_count_list:
        id_1 = char_to_id[char_1]
        id_2 = char_to_id[char_2]
        N[id_1, id_2] = n
    return N


def plot(N: torch.Tensor, id_to_char: dict):
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    n_char = len(id_to_char)

    for i in range(n_char):
        for j in range(n_char):
            chstr = id_to_char[i] + id_to_char[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")


def get_matrix_prob(N: torch.Tensor) -> torch.Tensor:
    P = copy(N).float()
    P /= P.sum(dim=1, keepdims=True)
    return P


def build_dataset(bigram_list: list[Tuple[str, str]], char_to_id: dict):
    xs, ys = [], []
    for ch1, ch2 in bigram_list:
        xs.append(char_to_id[ch1])
        ys.append(char_to_id[ch2])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    return xs, ys


def read(filename: str) -> List[str]:
    with open(filename) as f:
        return seq(f.readlines()).map(str.strip).list()
