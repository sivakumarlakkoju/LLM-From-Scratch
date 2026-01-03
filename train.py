import torch
import tiktoken
import torch.nn as nn
from GPT import GPT

torch.manual_seed(32)


def generate_text(model, idx, max_new_tokens, context):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context:]
        with torch.no_grad():
            logits = model(idx_cond)

        print(logits.shape)
        logits = logits[:, -1, :]
        print(logits.shape)
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    vec = token_ids.squeeze(0)
    return tokenizer.decode(vec.tolist())


def text_generation_loss(
GPT2_CUSTOM = {
        "vocab_size": 50257,
        "context": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 6,
        "dropout": 0.1,
        "qkv_bias": False
        }

GPT2_SMALL = {
        "vocab_size": 50257,
        "context": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "dropout": 0.1,
        "qkv_bias": False
        }

GPT2_MEDIUM = {
        "vocab_size": 50257,
        "context": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "dropout": 0.1,
        "qkv_bias": False
        }

GPT2_LARGE = {
        "vocab_size": 50257,
        "context": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "dropout": 0.1,
        "qkv_bias": False
        }

GPT2_XL = {
        "vocab_size": 50257,
        "context": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 40,
        "dropout": 0.1,
        "qkv_bias": False
        }

model = GPT(GPT2_CUSTOM)
#model.eval()


sample_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")


token_ids = generate_text(
        model=model,
        idx=text_to_token_ids(sample_context, tokenizer),
        max_new_tokens = 10,
        context=GPT2_CUSTOM['context']
        )

print(token_ids_to_text(token_ids, tokenizer))
