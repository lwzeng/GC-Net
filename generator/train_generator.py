from __future__ import print_function
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from generateData import *
from modul import *
import argparse
from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
Save_path = "/NAS_REMOTE/weizeng/model/weakdetection/coord2mask/"
parser = argparse.ArgumentParser(
    description='coord to map')
parser.add_argument('--mode', default="rotaRectangle", help='rectangle, rotaRectangle, rotaEllipse, quadrangle')
parser.add_argument('--loss_mode', default="diceLoss", help='loss model:crossEntropy, diceLoss')
parser.add_argument('--workers', default=1, type=int, help='worker number')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
# parser.add_argument('--resume', default="weights/coord2map20000.pth", type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--nc', default=1, type=int, help='Number of channels in the output images.')
parser.add_argument('--nz', default=1, type=int, help='Number of channels in the input images.')
parser.add_argument('--ngf', default=32, type=int, help='Size of feature maps in generator.')
parser.add_argument('--iteration', default=120000, type=int, help='Number of training iteration.')
parser.add_argument('--bbox_min', default=0.05, type=float, help='The min size of generate bbox.')
parser.add_argument('--bbox_max', default=0.95, type=float, help='The max size of generate bbox.')
parser.add_argument('--in_size', default=12, type=int, help='The intput size of generater.')
parser.add_argument('--out_size', default=224, type=int, help='The output size of generater.')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for optimizers.')
parser.add_argument('--beta1', default=0.5, type=float, help='Beta1 hyperparam for Adam optimizers.')
parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs available. Use 0 for CPU mode.')

parser.add_argument('--save_folder', default=Save_path + parser.parse_args().mode+"/45degree/", help='Directory for saving checkpoint models')
parser.add_argument('--log_folder', default='logs/'+parser.parse_args().mode+"/45degree/", help='Directory for saving log of tensorboard')
args = parser.parse_args()


lr_steps = [20000, 40000, 80000, 100000]



if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.log_folder):
    os.mkdir(args.log_folder)

logwriter = SummaryWriter(args.log_folder)


class diceloss(torch.nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

# Create the dataset
dataset = generate_data(args.bbox_min, args.bbox_max, args.in_size, args.out_size, args.iteration * args.batch_size, args.mode)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Create the generator
if args.mode == "rectangle":
    netG = rectGenerator(args.nz, args.nc, args.ngf).to("cuda")
elif args.mode == "rotaRectangle" or args.mode == "rotaEllipse":
    netG = rotaGenerator(args.nz, args.nc, args.ngf).to("cuda")
elif args.mode == "quadrangle":
    netG = quadGenerator(args.nz, args.nc, args.ngf).to("cuda")

# Handle multi-gpu if desired
if args.ngpu > 1:
    netG = nn.DataParallel(netG, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    netG.load_weights(args.resume)
else:
    netG.apply(weights_init)
# optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

optimizerG = optim.SGD([{'params': netG.parameters()}],
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training Loop
print("Starting Training", len(dataloader), "iteration....")
# For each epoch
total_loss = 0
step_index = 0
if args.loss_mode == "crossEntropy":
    criterion = nn.BCEWithLogitsLoss()
elif args.loss_mode == "diceLoss":
    criterion = diceloss()
netG.train()
for index, (img, gt) in enumerate(dataloader, 0):
    iter_index = args.start_iter + index
    img = img.cuda()
    # print(img.shape)
    gt = gt.cuda()

    # print(img.shape, gt.shape)
    netG.zero_grad()
    out = netG(img)
    loss = criterion(out, gt)
    loss.backward()
    optimizerG.step()

    total_loss += loss.item()
    if iter_index % 20 == 0:
        if iter_index == args.start_iter:
            total_loss = total_loss
        else:
            total_loss = total_loss / 20
        print("args.lr:", args.lr, "iter_index:", iter_index, "loss:", total_loss)
        logwriter.add_scalar('Train/Loss', total_loss, iter_index)
        total_loss = 0
    if iter_index in lr_steps:
        args.lr = args.lr * 0.1
        for param_group in optimizerG.param_groups:
            param_group['lr'] = args.lr

    if iter_index % 5000 == 0:
        torch.save(netG.state_dict(), args.save_folder +'/coord2map_'+ str(iter_index) + '.pth')
torch.save(netG.state_dict(), args.save_folder +'/coord2map_'+ str(iter_index) + '.pth')