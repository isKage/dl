from torch import nn


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        x = self.module(x)
        return x
