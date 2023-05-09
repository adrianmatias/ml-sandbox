from pprint import pprint

import data
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import MultiheadAttention
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
This script is based on the lesson
"Let's build GPT: from scratch, in code, spelled out."
https://www.youtube.com/watch?v=kCc8FmEb1nY

It justifies the convenience of transformer architecture to address the modeling of sequences of tokens.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Conf:
    batch_size = 64
    seq_len = 256
    embedding_dim = 384
    num_heads = 4
    dropout = 0.2


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

    model = TransformerAutoRegressive(vocab_size=tokenizer.vocab_size)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    loss_list = []
    step_count = 10
    epoch_count = 1

    for epoch in range(epoch_count):
        for i in range(step_count):
            x, y = dataset.get_batch()
            logits, loss = model(x, y)
            loss_list.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step(metrics=loss)

        print(epoch, loss)
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


class TransformerAutoRegressive(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=Conf.embedding_dim
        )
        self.positional_encoding = nn.Embedding(
            num_embeddings=Conf.seq_len, embedding_dim=Conf.embedding_dim
        )
        block_count = 2
        self.block_seq = nn.Sequential(*[Block() for _ in range(block_count)])
        self.normalization = nn.LayerNorm(normalized_shape=Conf.embedding_dim)
        self.linear = nn.Linear(in_features=Conf.embedding_dim, out_features=vocab_size)

    def forward(self, x, y):
        x = self.embedding(x) + self.positional_encoding(torch.arange(Conf.seq_len))
        x = self.block_seq(x)
        x = self.normalization(x)
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)

        if y is None:
            loss = None
        else:
            batch, seq_len, emb_dim = probs.shape
            loss = F.cross_entropy(
                input=probs.view(batch * seq_len, emb_dim),
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


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.multi_head_attention = MultiheadAttention(
            embed_dim=Conf.embedding_dim, num_heads=Conf.num_heads, dropout=Conf.dropout
        )
        self.normalization = nn.LayerNorm(normalized_shape=Conf.embedding_dim)
        self.ffn = FeedforwardNeuralNetModel(input_dim=Conf.embedding_dim)

    def forward(self, x):
        residual = x
        attn_output, attn_output_weights = self.multi_head_attention(
            query=x, key=x, value=x
        )
        x = residual + self.normalization(attn_output)
        x = self.ffn(x)
        return x


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden_dim = input_dim * 4

        self.layer_seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(normalized_shape=Conf.embedding_dim),
        )

    def forward(self, x):
        residual = x
        return residual + self.layer_seq(x)


if __name__ == "__main__":
    main()
