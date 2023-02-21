from pprint import pprint

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import data

"""
This script is based on the lesson
"Let's build GPT: from scratch, in code, spelled out."
https://www.youtube.com/watch?v=kCc8FmEb1nY

It justifies the convenience of transformer architecture to address the modeling of sequences of tokens.
"""


class Conf:
    batch_size = 32
    seq_len = 8
    embedding_dim = 2


def main():
    pprint(Conf.__dict__)

    is_shakespeare = True

    if is_shakespeare:
        dataset, tokenizer = data.get_shakespeare(
            batch_size=Conf.batch_size, seq_len=Conf.seq_len
        )
    else:
        dataset, tokenizer = data.build_data_random(
            text_size=1000, batch_size=Conf.batch_size, seq_len=Conf.seq_len
        )

    print(f"len(dataset): {len(dataset)}")
    print(f"tokenizer.vocab_size: {tokenizer.vocab_size}")

    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    loss_list = []
    max_step_count = 1000
    for epoch in range(20):
        for i in range(max_step_count):
            x, y = dataset.get_batch()
            logits, loss = model(x, y)
            loss_list.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step(metrics=loss)

    print(loss)
    print(logits.shape)

    is_plot = True
    if is_plot:
        loss_moving_average = torch.tensor(loss_list).view(-1, 100).mean(dim=1)
        plt.plot(loss_moving_average)
        plt.show()

    context = torch.zeros((1, 1), dtype=torch.long)
    generated_token_list = model.generate(x=context, token_count=100)
    print(generated_token_list)

    print(tokenizer.tokenize_back(id_list=generated_token_list.tolist()[0]))


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, x, y):
        logits = self.embedding(x)

        if y is None:
            loss = None
        else:
            batch, seq_len, emb_dim = logits.shape
            loss = F.cross_entropy(
                input=logits.view(batch * seq_len, emb_dim),
                target=y.view(batch * seq_len),
            )
        return logits, loss

    def generate(self, x, token_count):
        for _ in range(token_count):
            logits, _ = self(x=x, y=None)

            probs = F.softmax(logits[:, -1, :], dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x


if __name__ == "__main__":
    main()
