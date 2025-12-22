import torch
import torch.nn as nn

torch.manual_seed(22)


class GELU(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):

        return 0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x + 0.044715*torch.pow(x, 3))))


class NNSkip(nn.Module):
    def __init__(self, emb_dim, use_skip):
        super().__init__()
        self.use_skip = use_skip
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(emb_dim, emb_dim), GELU()) for _ in range(20)])
        self.output = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
            if self.use_skip and x.shape == y.shape:
                x = x + y
            else:
                x = y
        return self.output(x)


model1 = NNSkip(3, True)
model2 = NNSkip(3, False)


def print_gradients(model, x):
    output = model(x)

    target = torch.tensor([0.])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

sample_input = torch.randn(3,)
print_gradients(model1, sample_input)
print_gradients(model2, sample_input)
