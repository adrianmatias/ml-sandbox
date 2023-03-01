import torch
from torch import nn
from torch.nn import MultiheadAttention

from bigram_language_model import BigramLanguageModel


class BigramLanguageModelTorchLayers(BigramLanguageModel):
    """
    This class implements a neural network based on the lecture
    https://www.youtube.com/watch?v=kCc8FmEb1nY
    Let's build GPT: from scratch, in code, spelled out. by Andrej Karpathy.
    The objective is to assimilate the contents of the lecture and practice with
    pytorch, transformer architecture, and neural network training.
    This is approached by implementing a gpt-like model from scratch using the
    layers pytorch provides, instead of the custom ones from the lecture,
    to replicate the results of loss and text generation.
    """

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
        self.block_size = block_size
        self.device = device
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=number_embeddings
        )
        self.positional_encoding = nn.Embedding(
            num_embeddings=block_size, embedding_dim=number_embeddings
        )
        self.block_seq = nn.Sequential(
            *[
                Block(
                    embed_dim=number_embeddings, num_heads=number_heads, dropout=dropout
                )
                for _ in range(number_layers)
            ]
        )
        self.normalization = nn.LayerNorm(normalized_shape=number_embeddings)
        self.linear = nn.Linear(in_features=number_embeddings, out_features=vocab_size)

    def forward(self, contexts: torch.Tensor, targets=None):
        batch_size, seq_size = contexts.shape
        x = self.embedding(contexts) + self.positional_encoding(
            torch.arange(seq_size, device=self.device)
        )
        x = self.block_seq(x)
        x = self.normalization(x)
        logits = self.linear(x)

        if targets is None:
            loss = None
        else:
            batch_size, seq_size, emb_size = logits.shape
            loss = torch.nn.functional.cross_entropy(
                input=logits.view(batch_size * seq_size, emb_size),
                target=targets.view(batch_size * seq_size),
            )
        return logits, loss


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.multi_head_attention = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=False
        )
        self.normalization_mha = nn.LayerNorm(normalized_shape=embed_dim)
        self.ffw = FeedforwardNeuralNetModel(input_dim=embed_dim)
        self.normalization_ffw = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        residual = x
        x = self.normalization_mha(x)
        attn_output, attn_output_weights = self.multi_head_attention(
            query=x, key=x, value=x
        )
        x = residual + self.normalization_mha(attn_output)
        residual = x
        x = residual + self.ffw(self.normalization_ffw(x))
        return x


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden_dim = input_dim * 4
        self.layer_seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(normalized_shape=input_dim),
        )

    def forward(self, x):
        return self.layer_seq(x)
