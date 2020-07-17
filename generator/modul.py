import torch.nn as nn


# Generator Code
class rectGenerator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(rectGenerator, self).__init__()
        self.vect2Map = nn.Sequential(
            nn.Linear(4, 144),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.UpsamplingNearest2d(scale_factor=2),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64

        )

    def forward(self, input):
        x = self.vect2Map(input)
        x = x.view(x.size(0), 1, 12, 12)
        # print("self.fc2(x):", x.shape)
        return self.main(x)

# Generator Code
class quadGenerator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(quadGenerator, self).__init__()
        self.vect2Map = nn.Sequential(
            nn.Linear(8, 144),
            nn.ReLU(True),
            nn.Linear(144, 676),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # # # state size. (ngf*2) x 16 x 16
            # nn.UpsamplingNearest2d(scale_factor=2),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64

        )

    def forward(self, input):
        x = self.vect2Map(input)
        # x = x.view(x.size(0), 1, 12, 12)
        x = x.view(x.size(0), 1, 26, 26)
        # print("self.fc2(x):", x.shape)
        return self.main(x)

# Generator Code
class rotaGenerator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(rotaGenerator, self).__init__()
        self.vect2Map = nn.Sequential(
            nn.Linear(5, 144),
            nn.ReLU(True),
            # nn.Linear(144, 676),
            # nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # # state size. (ngf*2) x 16 x 16
            nn.UpsamplingNearest2d(scale_factor=2),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64

        )

    def forward(self, input):
        x = self.vect2Map(input)
        # x = x.view(x.size(0), 1, 12, 12)
        x = x.view(x.size(0), 1, 12, 12)
        # print("self.fc2(x):", x.shape)
        return self.main(x)