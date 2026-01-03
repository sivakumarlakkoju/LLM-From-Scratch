import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(32)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context"], config["emb_dim"])
        self.dropout = nn.Dropout(config["dropout"])
        self.transformerlayer = nn.Sequential(
                *[TransformerLayer(config)
                    for _ in range(config["n_layers"])]
                )
        self.norm = LayerNorm(config)
        self.out_head = nn.Linear(
                config["emb_dim"],
                config["vocab_size"],
                bias=False
                )

    def forward(self, in_):
        batch_size, seq_len = in_.shape
        x = self.token_emb(in_)
        x += self.pos_emb(torch.arange(seq_len, device=in_.device))
        x = self.dropout(x)
        x = self.transformerlayer(x)
        x = self.norm(x)
        logits = self.out_head(x)

        return logits


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (config["emb_dim"] % config["n_heads"] == 0)
        self.head_dim = config["emb_dim"] // config["n_heads"]
        self.n_heads = config["n_heads"]
        self.W_q = nn.Linear(
                config["emb_dim"], config["emb_dim"], bias=config["qkv_bias"]
                )
        self.W_k = nn.Linear(
                config["emb_dim"], config["emb_dim"], bias=config["qkv_bias"]
                )
        self.W_v = nn.Linear(
                config["emb_dim"], config["emb_dim"], bias=config["qkv_bias"]
                )
        self.dropout = nn.Dropout(config["dropout"])
        self.context_out = nn.Linear(config["emb_dim"], config["emb_dim"])
        self.register_buffer(
                'mask',
                torch.triu(
                    torch.ones(config["context"], config["context"]),
                    diagonal=1
                    )
                )

    def forward(self, x):
        B, C, E = x.shape
        q = self.W_q(x)  # (B, C, Emb)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.view(B, C, self.n_heads, self.head_dim)
        k = k.view(B, C, self.n_heads, self.head_dim)
        v = v.view(B, C, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # attention scores
        attn_scores = q @ k.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:C, :C], -torch.inf)

        attn_weights = torch.softmax(attn_scores / E**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ v).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
                B, C, E
                )

        context_vec = self.context_out(context_vec)
        return context_vec


# LN, ATTN, Drop, SKIP, LN, FFN, Drop, Skip
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LN1 = LayerNorm(config)
        self.multihead_attn = MultiHeadAttention(config)
        self.dropout = nn.Dropout(config["dropout"])
        self.LN2 = LayerNorm(config)
        self.FFN = PFFN(config)

    def forward(self, x):
        y = self.LN1(x)
        y = self.multihead_attn(y)
        y = self.dropout(y)
        x = x + y
        y = self.LN2(x)
        y = self.FFN(y)
        y = self.dropout(y)
        x = x + y

        return x


class LayerNorm(nn.Module):
    def __init__(self, config, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(config["emb_dim"]))
        self.shift = nn.Parameter(torch.ones(config["emb_dim"]))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=True)
        z = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * z + self.shift


class GELU(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(
                1 + torch.tanh(
                    torch.sqrt(
                        torch.tensor(2.0/torch.pi)
                        )*(x + 0.044715*torch.pow(x, 3))
                    )
                )


class PFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
                GELU(),
                nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
                )

    def forward(self, x):
        return self.layers(x)
