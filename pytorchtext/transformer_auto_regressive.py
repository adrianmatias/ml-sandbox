import torch
from torch import nn
from torch.nn import Sequential


class TransformerAutoRegressive(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        seq_len: int,
        attention_head_count: int,
    ):
        super(TransformerAutoRegressive, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        # nn.Positional()
        # self.pos_encoding = nn.Embedding(
        #     num_embeddings=seq_len, embedding_dim=embedding_dim
        # )
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=attention_head_count,
        )
        self.feed_forward = FeedForwardNet(
            in_dim=embedding_dim, hidden_dim=vocab_size * 8, out_dim=embedding_dim
        )
        self.linear = nn.Linear(
            in_features=embedding_dim, out_features=vocab_size, bias=False
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.embedding(x)
        # emb_y = self.embedding(y)
        # print(x)
        batch_len, seq_len, x_pos_dim = x.shape
        # print(batch_len, x_pos_dim)
        # x_pos = torch.arange(batch_len)
        # pos_enc = self.pos_encoding(x_pos)
        # out = self.multi_head_attention(embedding)
        attn_output, attn_output_weights = self.multi_head_attention(
            query=x, key=x, value=x
        )
        out = self.feed_forward(attn_output.view(batch_len, seq_len, -1))

        # out = self.feed_forward(attn_output.view(batch_len, seq_len, -1))
        out = self.linear(out)
        # out_max = torch.sum(out, dim=1)
        out = self.softmax(out)
        return out

    def evaluate(self, x, y):
        x = self.embedding(x)
        y = self.embedding(y)


class FeedForwardNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(FeedForwardNet, self).__init__()
        self.layer_seq = Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim, out_features=out_dim),
        )

    def forward(self, x):
        return self.layer_seq(x)
