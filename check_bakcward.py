import torch
import torch.nn as nn
from torch.optim import Adam

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Wa = nn.Parameter(torch.ones(1).double(), requires_grad=True)
        
        main = nn.Sequential(
            nn.LeakyReLU(1),
        )
        self.main = main

    def forward(self, inputs):
        inputs = inputs * self.Wa
        out = self.main(inputs)
        return out.view(-1)

discriminator = Discriminator()

optim_disc = Adam(discriminator.parameters(), weight_decay=0, lr=0.5)



for i in range(10):
    discriminator.zero_grad()
    X = torch.tensor([1.]).double()
    loss = discriminator(X).mean(0).view(1)
    print(loss)
    loss.backward()
    optim_disc.step()
