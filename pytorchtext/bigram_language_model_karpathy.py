import shutil
import time
from typing import Tuple

import torch
from bigram_language_model import BigramLanguageModel


class BigramLanguageModelKarpathy(BigramLanguageModel):
    def __init__(
        self,
        vocab_size: int,
        number_embeddings: int,
        block_size: int,
        number_heads: int,
        number_layers: int,
        dropout: float,
        device,
    ) -> None:
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.block_size = block_size
        self.token_embedding_table = torch.nn.Embedding(vocab_size, number_embeddings)
        self.position_embedding_table = torch.nn.Embedding(
            block_size, number_embeddings
        )
        self.blocks = torch.nn.Sequential(
            *[
                Block(number_embeddings, number_heads, block_size, dropout)
                for _ in range(number_layers)
            ]
        )
        self.final_layer_norm = torch.nn.LayerNorm(number_embeddings)
        self.language_model_head = torch.nn.Linear(number_embeddings, vocab_size)
        self.device = device

    def forward(self, contexts: torch.Tensor, targets=None):
        B, T = contexts.shape
        # index_x and targets are both (batch_size, block_size)
        token_embeddings = self.token_embedding_table(contexts)  # (B, T, C)
        position_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.final_layer_norm(x)  # (B, T, C)
        logits = self.language_model_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape

        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss


class Head(torch.nn.Module):
    """One head of self-attention"""

    def __init__(
        self, head_size: int, block_size: int, number_embeddings: int, dropout: float
    ) -> None:
        super().__init__()
        self.key = torch.nn.Linear(number_embeddings, head_size, bias=False)
        self.query = torch.nn.Linear(number_embeddings, head_size, bias=False)
        self.value = torch.nn.Linear(number_embeddings, head_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # Attention scores
        weights = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) ----> (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = torch.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # Weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        out = weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(torch.nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(
        self,
        number_heads: int,
        head_size: int,
        block_size: int,
        number_embeddings: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                Head(head_size, block_size, number_embeddings, dropout)
                for _ in range(number_heads)
            ]
        )
        self.projection = torch.nn.Linear(number_embeddings, number_embeddings)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.dropout(self.projection(output))
        return output


class FeedForward(torch.nn.Module):
    """A simple feed-forward layer."""

    def __init__(self, number_embeddings: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(number_embeddings, 4 * number_embeddings),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * number_embeddings, number_embeddings),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(torch.nn.Module):
    """Transformer block communication followed by computation."""

    def __init__(
        self, number_embeddings: int, number_heads: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        assert number_embeddings % number_heads == 0
        head_size = number_embeddings // number_heads
        self.self_attention = MultiHeadAttention(
            number_heads, head_size, block_size, number_embeddings, dropout
        )
        self.feed_forward = FeedForward(number_embeddings, dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(number_embeddings)
        self.layer_norm_2 = torch.nn.LayerNorm(number_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


# Data loading
def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else validation_data
    indices_start = torch.randint(len(data) - block_size - 1, (batch_size,))
    contexts = torch.stack([data[i : i + block_size] for i in indices_start])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in indices_start])
    return contexts, targets


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> dict:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            contexts, targets = get_batch(
                split, block_size, batch_size, train_data, validation_data
            )
            _, loss = model(contexts, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
