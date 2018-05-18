import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.z_dim = args.z_dim
        self.m_g = args.m_g  # Dataset Dependent
        self.ch = args.ngf

        self.linear = nn.Linear(self.z_dim, self.m_g*self.m_g*self.ch)
        self.activation = nn.ReLU()
        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(self.ch, self.ch//2, 4, 2, 1),
            nn.BatchNorm2d(self.ch//2),
            nn.ReLU(),

            nn.ConvTranspose2d(self.ch//2, self.ch//4, 4, 2, 1),
            nn.BatchNorm2d(self.ch//4),
            nn.ReLU(),

            nn.ConvTranspose2d(self.ch//4, self.ch//8, 4, 2, 1),
            nn.BatchNorm2d(self.ch//8),
            nn.ReLU(),

            nn.ConvTranspose2d(self.ch//8, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.activation(self.linear(z))
        out = out.view(-1, self.ch, self.m_g, self.m_g)
        out = self.deconv(out)

        return out


def test_generator():
    class Args:
        z_dim = 128
        m_g = 4
        ngf = 512

    args = Args()
    gen = Generator(args)
    z = Variable(torch.randn(10, 128))

    out = gen(z)
    print(out.shape)


if __name__ == "__main__":
    test_generator()
