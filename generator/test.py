from __future__ import print_function
import os
import torch.utils.data
import torch.nn.functional as F
from generateData import *
from modul import *
import argparse
import numpy as np
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

Save_path = "/NAS_REMOTE/weizeng/model/weakdetection/"


parser = argparse.ArgumentParser(
    description='coord to map')
parser.add_argument('--mode', default="rotaRectangle", help='rectangle, rotaRectangle, rotaEllipse, quadrangle')
parser.add_argument('--workers', default=1, type=int, help='worker number')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--back_bone', default="vgg16", help='backbone type')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--nc', default=1, type=int, help='Number of channels in the output images.')
parser.add_argument('--nz', default=1, type=int, help='Number of channels in the input images.')
parser.add_argument('--ngf', default=32, type=int, help='Size of feature maps in generator.')
parser.add_argument('--iteration', default=128, type=int, help='Number of training iteration.')
parser.add_argument('--bbox_min', default=0.05, type=float, help='The min size of generate bbox.')
parser.add_argument('--bbox_max', default=0.95, type=float, help='The max size of generate bbox.')
parser.add_argument('--in_size', default=12, type=int, help='The intput size of generater.')
parser.add_argument('--out_size', default=224, type=int, help='The output size of generater.')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for optimizers.')
parser.add_argument('--beta1', default=0.5, type=float, help='Beta1 hyperparam for Adam optimizers.')
parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs available. Use 0 for CPU mode.')

parser.add_argument('--trained_model', default=Save_path+"/coord2mask/"+parser.parse_args().mode+'/45degree/coord2map_119999.pth', help='Directory for saving checkpoint models')
args = parser.parse_args()


# Create the dataset
dataset = generate_data(args.bbox_min, args.bbox_max, args.in_size, args.out_size, args.iteration * args.batch_size, args.mode)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

# Create the generator
# Create the generator
if args.mode == "rectangle":
    netG = rectGenerator(args.nz, args.nc, args.ngf).to("cuda")
elif args.mode == "rotaRectangle" or args.mode == "rotaEllipse":
    netG = rotaGenerator(args.nz, args.nc, args.ngf).to("cuda")
elif args.mode == "quadrangle":
    netG = quadGenerator(args.nz, args.nc, args.ngf).to("cuda")
netG.load_state_dict(torch.load(args.trained_model))

# Training Loop
print("Starting Training", len(dataloader), "iteration....")
# For each epoch
total_loss = 0
netG.eval()
for iter_index, (img, gt) in enumerate(dataloader, 0):
    img = img.cuda()
    print("img:",img.shape)
    gt = gt.cuda()
    out = netG(img)
    out = F.sigmoid(out)
    loss = F.binary_cross_entropy(out, gt)

    print(out.shape, gt.shape)
    print(loss.item())

    gt = gt.cpu().numpy().reshape([args.batch_size, args.out_size, args.out_size])
    out = out.cpu().detach().numpy().reshape([args.batch_size, args.out_size, args.out_size])
    # out[np.where(out >= 0.5)] = 1
    # out[np.where(out < 0.5)] = 0


    for i in range(out.shape[0]):
        cv2.imwrite("temp/"+str(iter_index)+"_"+str(i)+"_out.jpg", out[i, :, :]*255)
        cv2.imwrite("temp/"+str(iter_index)+"_"+str(i)+"_gt.jpg", gt[i, :, :]*255)


    # gt = np.array(gt.item()).reshape([args.out_size, args.out_size])
    # out = np.array(out.item()).reshape([args.out_size, args.out_size])


    # gt = gt.cpu().numpy().reshape([args.out_size, args.out_size])
    # out = out.cpu().detach().numpy().reshape([args.out_size, args.out_size])
    # out[np.where(out >= 0.5)] = 1
    # out[np.where(out < 0.5)] = 0
    #
    # #visualization
    # cv2.imwrite("temp/"+str(iter_index)+"_gt.jpg", gt*255)
    # cv2.imwrite("temp/"+str(iter_index)+"_out.jpg", out*255)
    # print(loss.item())