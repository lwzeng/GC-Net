import torch
import numpy as np
import torch.nn as tnn
from matplotlib import pyplot as plt
import cv2


#angle -90 to 90
class gene_rotaEllipse(torch.nn.Module):
    def __init__(self, img_h, img_w):
        super(gene_rotaEllipse, self).__init__()
        self.imgh = img_h
        self.imgw = img_w
        with torch.no_grad():
            self.grid_h, self.grid_w = torch.meshgrid([torch.linspace(0, 1, img_h), torch.linspace(0, 1, img_w)])
            self.grid_h = self.grid_h.view(1, self.imgw, self.imgh)
            self.grid_w = self.grid_w.view(1, self.imgw, self.imgh)
            self.PI = torch.tensor(np.pi)
        self.smooth = tnn.Parameter(torch.tensor(10.0, dtype=torch.float32, requires_grad=True))
        # print(self.grid_w, self.grid_h)

    def forward(self, center_x, center_y, axis_a, axis_b, angle):
        batch_size = center_x.shape[0]
        self.grid_h, self.grid_w, self.PI = self.grid_h.cuda(), self.grid_w.cuda(), self.PI.cuda()
        grid_w = self.grid_w.expand(batch_size, self.imgw, self.imgh)
        grid_h = self.grid_h.expand(batch_size, self.imgw, self.imgh)
        center_x = center_x.expand(batch_size, self.imgh*self.imgw)
        # center_x = center_x.view(batch_size, self.imgw, self.imgh)*self.imgw
        center_x = center_x.view(batch_size, self.imgw, self.imgh)

        center_y = center_y.expand(batch_size, self.imgh*self.imgw)
        # center_y = center_y.view(batch_size, self.imgw, self.imgh)*self.imgh
        center_y = center_y.view(batch_size, self.imgw, self.imgh)

        axis_a = axis_a.expand(batch_size, self.imgh*self.imgw)
        # axis_a = axis_a.view(batch_size, self.imgw, self.imgh)*self.imgw
        axis_a = axis_a.view(batch_size, self.imgw, self.imgh)

        axis_b = axis_b.expand(batch_size, self.imgh*self.imgw)
        # axis_b = axis_b.view(batch_size, self.imgw, self.imgh)*self.imgw
        axis_b = axis_b.view(batch_size, self.imgw, self.imgh)

        angle = angle.expand(batch_size, self.imgh*self.imgw)
        angle = angle.view(batch_size, self.imgw, self.imgh)

        # print(angle.dtype, self.PI.dtype, self.grid_w.dtype, center_x.dtype)
        # rotated ellipse function
        part1_temp = ((grid_w - center_x)*torch.cos(self.PI * angle)) + ((grid_h - center_y)*torch.sin(self.PI * angle))
        part1 = part1_temp * part1_temp
        part2_temp = ((grid_w - center_x)*torch.sin(self.PI * angle)) - ((grid_h - center_y)*torch.cos(self.PI * angle))
        part2 = part2_temp * part2_temp
        input = part1 / (axis_a * axis_a) + part2 / (axis_b * axis_b)
        image = 0.5 + (1 / self.PI) * torch.atan(self.smooth*input - self.smooth)
        return 1-image.view(batch_size, 1, image.shape[1], image.shape[2])

# c2mc =  C2MC_rotaEllipse(img_h=224, img_w= 224)
#
# center_x, center_y, axis_a, axis_b, angle = torch.tensor([[0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.3], [0.3]], dtype=torch.float),\
#                                             torch.tensor([[0.2], [0.2]], dtype=torch.float),\
#                                             torch.tensor([[0.25], [0.25]], dtype=torch.float)
#
# image = c2mc(center_x, center_y, axis_a, axis_b, angle)
# print(image.shape)
# print(image[0][0][30][200].detach().numpy()*255)
# plt.imshow((image[0][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# cv2.imwrite('Figure_1.png', (image[0][0].detach().numpy())*255)
# plt.imshow((image[1][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()




#angle -90 to 90
class gene_rotaPrism(torch.nn.Module):
    def __init__(self, img_h, img_w):
        super(gene_rotaPrism, self).__init__()
        self.imgh = img_h
        self.imgw = img_w
        with torch.no_grad():
            self.grid_h, self.grid_w = torch.meshgrid([torch.linspace(0, 1, img_h), torch.linspace(0, 1, img_w)])
            self.grid_h = self.grid_h.view(1, self.imgw, self.imgh)
            self.grid_w = self.grid_w.view(1, self.imgw, self.imgh)
            self.PI = torch.tensor(np.pi)
            self.exp = torch.tensor(10.0)
        self.smooth = tnn.Parameter(torch.tensor(50.0, dtype=torch.float32, requires_grad=True))

    def forward(self, center_x, center_y, axis_a, axis_b, angle):
        batch_size = center_x.shape[0]
        self.grid_h, self.grid_w, self.PI, self.exp = self.grid_h.cuda(), self.grid_w.cuda(), self.PI.cuda(), self.exp.cuda()
        grid_w = self.grid_w.expand(batch_size, self.imgw, self.imgh)
        grid_h = self.grid_h.expand(batch_size, self.imgw, self.imgh)
        center_x = center_x.expand(batch_size, self.imgh*self.imgw)
        center_x = center_x.view(batch_size, self.imgw, self.imgh)

        center_y = center_y.expand(batch_size, self.imgh*self.imgw)
        center_y = center_y.view(batch_size, self.imgw, self.imgh)

        axis_a = axis_a.expand(batch_size, self.imgh*self.imgw)
        axis_a = axis_a.view(batch_size, self.imgw, self.imgh)

        axis_b = axis_b.expand(batch_size, self.imgh*self.imgw)
        axis_b = axis_b.view(batch_size, self.imgw, self.imgh)

        angle = angle.expand(batch_size, self.imgh*self.imgw)
        angle = angle.view(batch_size, self.imgw, self.imgh)

        # rotated ellipse function
        part1_temp = ((grid_w - center_x)*torch.cos(self.PI * angle)) + ((grid_h - center_y)*torch.sin(self.PI * angle))
        part1 = part1_temp ** self.exp
        part2_temp = ((grid_w - center_x)*torch.sin(self.PI * angle)) - ((grid_h - center_y)*torch.cos(self.PI * angle))
        part2 = part2_temp ** self.exp
        input = part1 / (axis_a**self.exp) + part2 / (axis_b**self.exp)
        image = 0.5 + 1 / self.PI * torch.atan(self.smooth*input - self.smooth)
        return 1-image.view(batch_size, 1, image.shape[1], image.shape[2])

# c2mc =  C2MC_rotaEllipse(img_h=224, img_w= 224)
#
# center_x, center_y, axis_a, axis_b, angle = torch.tensor([[0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.3], [0.3]], dtype=torch.float),\
#                                             torch.tensor([[0.2], [0.2]], dtype=torch.float),\
#                                             torch.tensor([[0], [0.25]], dtype=torch.float)
#
# image = c2mc(center_x, center_y, axis_a, axis_b, angle)
# print(image.shape)
# plt.imshow((1-image[0][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[1][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()



#angle -90 to 90
class gene_rotaRectangle(torch.nn.Module):
    def __init__(self, img_h, img_w):
        super(gene_rotaRectangle, self).__init__()
        self.imgh = img_h
        self.imgw = img_w
        with torch.no_grad():
            self.grid_h, self.grid_w = torch.meshgrid([torch.linspace(0, 1, img_h), torch.linspace(0, 1, img_w)])
            self.grid_h = self.grid_h.view(1, self.imgw, self.imgh)
            self.grid_w = self.grid_w.view(1, self.imgw, self.imgh)
            self.PI = torch.tensor(np.pi)
            self.exp = torch.tensor(10.0)
        self.smooth = tnn.Parameter(torch.tensor(10.0, dtype=torch.float32, requires_grad=True))

    def forward(self, center_x, center_y, axis_a, axis_b, angle):
        batch_size = center_x.shape[0]
        self.grid_h, self.grid_w, self.PI, self.exp = self.grid_h.cuda(), self.grid_w.cuda(), self.PI.cuda(), self.exp.cuda()
        grid_w = self.grid_w.expand(batch_size, self.imgw, self.imgh)
        grid_h = self.grid_h.expand(batch_size, self.imgw, self.imgh)
        center_x = center_x.expand(batch_size, self.imgh*self.imgw)
        center_x = center_x.view(batch_size, self.imgw, self.imgh)

        center_y = center_y.expand(batch_size, self.imgh*self.imgw)
        center_y = center_y.view(batch_size, self.imgw, self.imgh)

        axis_a = axis_a.expand(batch_size, self.imgh*self.imgw)
        axis_a = axis_a.view(batch_size, self.imgw, self.imgh)

        axis_b = axis_b.expand(batch_size, self.imgh*self.imgw)
        axis_b = axis_b.view(batch_size, self.imgw, self.imgh)

        angle = angle.expand(batch_size, self.imgh*self.imgw)
        angle = angle.view(batch_size, self.imgw, self.imgh)

        # rotated rectangle function
        part1_temp = ((grid_w - center_x)*torch.cos(self.PI * angle)) \
                     - ((grid_h - center_y)*torch.sin(self.PI * angle))
        part1_temp = part1_temp/axis_a

        part2_temp = ((grid_w - center_x)*torch.sin(self.PI * angle)) \
                     + ((grid_h - center_y)*torch.cos(self.PI * angle))
        part2_temp = part2_temp/axis_b

        part1 = torch.abs(part1_temp + part2_temp)
        part2 = torch.abs(part1_temp - part2_temp)
        # part1 = (part1_temp + part2_temp)**2
        # part2 = (part1_temp - part2_temp)**2
        input = part1 + part2
        image = 0.5 + 1 / self.PI * torch.atan(self.smooth*input - self.smooth)
        return 1-image.view(batch_size, 1, image.shape[1], image.shape[2])
#
# c2mc =  C2MC_rotaRectangle(img_h=224, img_w= 224)
# center_x, center_y, axis_a, axis_b, angle = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.25], [0.25], [0.25]], dtype=torch.float),\
#                                             torch.tensor([[0], [0.25], [-0.25]], dtype=torch.float)
#
# image = c2mc(center_x, center_y, axis_a, axis_b, angle)
# print(image.shape)
# plt.imshow((1-image[0][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[1][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[2][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()


#angle -90 to 90
class gene_rotaRectangle(torch.nn.Module):
    def __init__(self, img_h, img_w):
        super(gene_rotaRectangle, self).__init__()
        self.imgh = img_h
        self.imgw = img_w
        with torch.no_grad():
            self.grid_h, self.grid_w = torch.meshgrid([torch.linspace(0, 1, img_h), torch.linspace(0, 1, img_w)])
            self.grid_h = self.grid_h.view(1, self.imgw, self.imgh)
            self.grid_w = self.grid_w.view(1, self.imgw, self.imgh)
            self.PI = torch.tensor(np.pi)
            self.exp = torch.tensor(10.0)
        self.smooth = tnn.Parameter(torch.tensor(10.0, dtype=torch.float32, requires_grad=True))

    def forward(self, center_x, center_y, axis_a, axis_b, angle):
        batch_size = center_x.shape[0]
        self.grid_h, self.grid_w, self.PI, self.exp = self.grid_h.cuda(), self.grid_w.cuda(), self.PI.cuda(), self.exp.cuda()
        grid_w = self.grid_w.expand(batch_size, self.imgw, self.imgh)
        grid_h = self.grid_h.expand(batch_size, self.imgw, self.imgh)
        center_x = center_x.expand(batch_size, self.imgh*self.imgw)
        center_x = center_x.view(batch_size, self.imgw, self.imgh)

        center_y = center_y.expand(batch_size, self.imgh*self.imgw)
        center_y = center_y.view(batch_size, self.imgw, self.imgh)

        axis_a = axis_a.expand(batch_size, self.imgh*self.imgw)
        axis_a = axis_a.view(batch_size, self.imgw, self.imgh)

        axis_b = axis_b.expand(batch_size, self.imgh*self.imgw)
        axis_b = axis_b.view(batch_size, self.imgw, self.imgh)

        angle = angle.expand(batch_size, self.imgh*self.imgw)
        angle = angle.view(batch_size, self.imgw, self.imgh)

        # rotated rectangle function
        part1_temp = ((grid_w - center_x)*torch.cos(self.PI * angle)) \
                     - ((grid_h - center_y)*torch.sin(self.PI * angle))
        part1_temp = part1_temp/axis_a

        part2_temp = ((grid_w - center_x)*torch.sin(self.PI * angle)) \
                     + ((grid_h - center_y)*torch.cos(self.PI * angle))
        part2_temp = part2_temp/axis_b

        part1 = torch.abs(part1_temp + part2_temp)
        part2 = torch.abs(part1_temp - part2_temp)
        # part1 = (part1_temp + part2_temp)**2
        # part2 = (part1_temp - part2_temp)**2
        input = part1 + part2
        image = 0.5 + 1 / self.PI * torch.atan(self.smooth*input - self.smooth)
        return 1-image.view(batch_size, 1, image.shape[1], image.shape[2])
#
# c2mc =  C2MC_rotaRectangle(img_h=224, img_w= 224)
# center_x, center_y, axis_a, axis_b, angle = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                             torch.tensor([[0.25], [0.25], [0.25]], dtype=torch.float),\
#                                             torch.tensor([[0], [0.25], [-0.25]], dtype=torch.float)
#
# image = c2mc(center_x, center_y, axis_a, axis_b, angle)
# print(image.shape)
# plt.imshow((1-image[0][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[1][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[2][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()


#angle -90 to 90
class gene_Rectangle(torch.nn.Module):
    def __init__(self, img_h, img_w):
        super(gene_Rectangle, self).__init__()
        self.imgh = img_h
        self.imgw = img_w
        with torch.no_grad():
            self.grid_h, self.grid_w = torch.meshgrid([torch.linspace(0, 1, img_h), torch.linspace(0, 1, img_w)])
            self.grid_h = self.grid_h.view(1, self.imgw, self.imgh)
            self.grid_w = self.grid_w.view(1, self.imgw, self.imgh)
            self.PI = torch.tensor(np.pi)
            self.exp = torch.tensor(10.0)
        self.smooth = tnn.Parameter(torch.tensor(10.0, dtype=torch.float32, requires_grad=True))

    def forward(self, center_x, center_y, axis_a, axis_b):
        batch_size = center_x.shape[0]
        self.grid_h, self.grid_w, self.PI, self.exp = self.grid_h.cuda(), self.grid_w.cuda(), self.PI.cuda(), self.exp.cuda()
        grid_w = self.grid_w.expand(batch_size, self.imgw, self.imgh)
        grid_h = self.grid_h.expand(batch_size, self.imgw, self.imgh)
        center_x = center_x.expand(batch_size, self.imgh*self.imgw)
        center_x = center_x.view(batch_size, self.imgw, self.imgh)

        center_y = center_y.expand(batch_size, self.imgh*self.imgw)
        center_y = center_y.view(batch_size, self.imgw, self.imgh)

        axis_a = axis_a.expand(batch_size, self.imgh*self.imgw)
        axis_a = axis_a.view(batch_size, self.imgw, self.imgh)

        axis_b = axis_b.expand(batch_size, self.imgh*self.imgw)
        axis_b = axis_b.view(batch_size, self.imgw, self.imgh)

        # rotated rectangle function
        part1 = ((grid_w - center_x)/axis_a + (grid_h - center_y)/axis_b)
        part2 = ((grid_w - center_x)/axis_a - (grid_h - center_y)/axis_b)

        part1 = torch.abs(part1)
        part2 = torch.abs(part2)

        input = part1 + part2
        image = 0.5 + 1 / self.PI * torch.atan(self.smooth*input - self.smooth)
        return 1-image.view(batch_size, 1, image.shape[1], image.shape[2])

# c2mc =  C2MC_Rectangle(img_h=224, img_w= 224)
# center_x, center_y, axis_a, axis_b = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                      torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float),\
#                                      torch.tensor([[0.5], [0.25], [0.25]], dtype=torch.float),\
#                                      torch.tensor([[0.25], [0.5], [0.25]], dtype=torch.float),\
#
# image = c2mc(center_x, center_y, axis_a, axis_b)
# print(image.shape)
# plt.imshow((1-image[0][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[1][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
# plt.imshow((1-image[2][0].detach().numpy())*255, cmap='gray', vmin=0, vmax=255)
# plt.show()
