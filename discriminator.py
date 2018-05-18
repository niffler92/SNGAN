import torch
import torch.nn as nn
from torch.autograd import Variable

from spectralnorm import SpectralNorm


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        m_g = args.m_g
        ch = args.ndf

        self.layer1 = self.make_layer(3, ch//8)
        self.layer2 = self.make_layer(ch//8, ch//4)
        self.layer3 = self.make_layer(ch//4, ch//2)
        self.layer4 = SpectralNorm(nn.Conv2d(ch//2, ch, 3, 1, 1), self.args)
        self.linear = SpectralNorm(nn.Linear(ch*m_g*m_g, 1), self.args)

    def make_layer(self, in_plane, out_plane):
        return nn.Sequential(
            SpectralNorm(
                nn.Conv2d(in_plane, out_plane, 3, 1, 1), self.args
            ),
            nn.LeakyReLU(0.1),
            SpectralNorm(
                nn.Conv2d(out_plane, out_plane, 4, 2, 1), self.args
            ),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out.squeeze()


def test_discriminator():
    class Args:
        sn = True
        ndf = 512
        m_g = 4
    args = Args()
    dis = Discriminator(args)
    x = Variable(torch.randn(10, 3, 32, 32))

    out = dis(x)
    print(out.shape)


if __name__ == "__main__":
    test_discriminator()
