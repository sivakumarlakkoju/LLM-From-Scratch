import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified Self-Attn
inputs = torch.rand((8, 3))

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# print(attn_scores_2)

# normalization of attn scores to get a sum 1
attn_weights_2 = attn_scores_2/attn_scores_2.sum()
# print(attn_weights_2)
# print(attn_weights_2.sum())

# def softmax_naive(x):
#    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print(attn_weights_2)
# print(attn_weights_2.sum())

context_vector_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vector_2 += attn_weights_2[i]*x_i
# print(context_vector_2)

attn_scores = torch.empty(8, 8)
# for loops are slow
# for i, x_i in enumerate(inputs):
#   for j, x_j in enumerate(inputs):
#       attn_scores[i,j] = torch.dot(x_i,x_j)

attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights.shape)
context_vectors = attn_weights @ inputs


# Self-Attn -- Scaled Dot Product Attn
# Further improvement with nn.Linear with bias=False, has optimized weight initialization
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, inputs):
        q = inputs @ self.W_q  # (B, C, Emb) @ (Emb, Emb)
        k = inputs @ self.W_k
        v = inputs @ self.W_v

        attn_scores = q @ k.T  # (B, C, Emb) @ (B, Emb, C)
        attn_weights = torch.softmax(
                attn_scores / k.shape[-1]**0.5, dim=-1
                )  # (B, C, C)

        context_vectors = attn_weights @ v  # (B, C, C) @ (B, C, Emb)
        return context_vectors


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attn_scores = q @ k.T
        attn_weights = torch.softmax(
                attn_scores / k.shape[-1]**0.5, dim=-1)
        return attn_weights @ v


# attn = SelfAttention(3, 3)
# print(attn(inputs))


# Causal Attn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.3, attn_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=attn_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=attn_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=attn_bias)
        self.dropout = nn.Dropout(dropout)
        # why register_buffer???
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

    def forward(self, x):
        _, c_len, _ = x.shape
        q = self.W_q(x)  # B, C, Emb
        k = self.W_k(x)  # B, C, Emb
        v = self.W_v(x)
        # transposing the C(1) and Emb(2) dimensions, keeping B(0) fixed
        attn_scores = q @ k.transpose(1, 2)  # B, C, C
        # pytorch ops with trailing underscore are inplace ops for e.g. masked_fill_()
        attn_scores.masked_fill_(self.mask.bool()[:c_len, :c_len], -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vectors = attn_weights @ v
        return context_vectors


# Causal Multi-Head Attn
class MultiHeadAttentionV1(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.3, attn_bias=False):
        super().__init__()
        d_out = d_out//num_heads
        self.heads = nn.ModuleList(
                [CausalAttention(d_in, d_out, context_length, dropout, attn_bias)
                 for _ in range(num_heads)]
                )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# More efficient with less matrix multiplications
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.3, attn_bias=False):
        super().__init__()
        # d_out should be divisible by num_heads
        assert (d_out % num_heads == 0)
        self.head_d_out = d_out // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_in, d_out, bias=attn_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=attn_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=attn_bias)
        self.dropout = nn.Dropout(dropout)
        self.context_out = nn.Linear(d_out, d_out)
        self.register_buffer(
                'mask',
                torch.tril(torch.ones(context_length, context_length), diagonal=1)
                )

    def forward(self, x):
        B, C, E = x.shape
        q = self.W_q(x)  # (B, C, Emb)
        k = self.W_k(x)
        v = self.W_v(x)
        
        q = q.view(B, C, self.num_heads, self.head_d_out)
        k = k.view(B, C, self.num_heads, self.head_d_out)
        v = v.view(B, C, self.num_heads, self.head_d_out) 
        
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


t = torch.randn(64, 1024, 768)

m = MultiHeadAttention(768, 768, 12, 1024)
print(m(t).shape)
