import torch
import torch.nn.functional as F


class NeuralNet:
    def __init__(self, n_char, seed=2147483647):
        self.n_char = n_char
        generator = torch.Generator().manual_seed(seed)
        self.W = torch.randn(
            (self.n_char, self.n_char), generator=generator, requires_grad=True
        )

    def get_probs(self, x):
        xenc = F.one_hot(
            x, num_classes=self.n_char
        ).float()  # input to the network: one-hot encoding
        logits = xenc @ self.W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        probs = counts / counts.sum(
            1, keepdims=True
        )  # probabilities for next character
        return probs

    def forward(self, x, y):
        x_size = x.nelement()
        reg_alpha = 0.1
        probs = self.get_probs(x=x)
        regularization = +reg_alpha * (self.W**2).mean()
        loss = -probs[torch.arange(x_size), y].log().mean() + regularization
        return loss

    def init_gradient(self):
        self.W.grad = None

    def fit(self, loss):
        self.init_gradient()
        loss.backward()
        learning_rate = 50
        self.W.data -= learning_rate * self.W.grad

    def sample(self, id_to_char: dict):
        g = torch.Generator().manual_seed(2147483647)

        n_samples = 5
        for i in range(n_samples):
            out = []
            ix = 0
            while True:
                probs = self.get_probs(x=torch.tensor([ix]))

                ix = torch.multinomial(
                    probs, num_samples=1, replacement=True, generator=g
                ).item()
                out.append(id_to_char[ix])
                if ix == 0:
                    break
            print("".join(out))
