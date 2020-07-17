import sys
sys.path.append("..")
import torch
import torch.nn as tnn
from torch.autograd import Variable
import os
import argparse
from tensorboardX import SummaryWriter
from img_loader import ImageList
import time
from module import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Save_path = "/path/to/model/"
parser = argparse.ArgumentParser(description='classifier for loss')
parser.add_argument('--dataset', default="ImageNet", help='CUB, ImageNet')
parser.add_argument('--loss', default="ObjBack", help='Obj, areaObj, ObjBack, areaObjBack')
if parser.parse_args().dataset == "CUB":
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--workers', default=16, type=int, help='worker number')
elif parser.parse_args().dataset == "ImageNet":
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--workers', default=16, type=int, help='worker number')
parser.add_argument('--back_bone', default="inceptionV3", help='backbone type:vgg16, googleNet, inceptionV3')
parser.add_argument('--generator_mode', default="CNN", help='math function or CNN')
parser.add_argument('--debug', default=True, type=bool, help='debug or not')
parser.add_argument('--class_backbone', default="inceptionV3", help='backbone type:vgg16, resnet50, googleNet, inceptionV3')
parser.add_argument('--Classifier', default=Save_path+parser.parse_args().dataset+"/classifier/"+parser.parse_args().class_backbone+"/"
                                            +"/cub200_199.pkl", help='backbone type')
parser.add_argument('--mask_shape', default="rotaEllipse", help='mask type:rectangle, rotaRectangle, rotaEllipse, quadrangle')
parser.add_argument('--nc', default=1, type=int, help='Number of channels in the output images.')
parser.add_argument('--nz', default=1, type=int, help='Number of channels in the input images.')
parser.add_argument('--ngf', default=32, type=int, help='Size of feature maps in generator.')
parser.add_argument('--generator', default=Save_path+"/coord2mask/"+parser.parse_args().mask_shape+"/45degree/coord2map_119999.pth", help='backbone type')
if parser.parse_args().dataset == "CUB":
    parser.add_argument('--data_root', default='/path/to/CUB_200_2011/', help='path of data root')
    parser.add_argument('--test_list', default='list/test.txt', help='path of test set list file')
    parser.add_argument('--train_list', default='list/train.txt', help='path of train set list file')
elif parser.parse_args().dataset == "ImageNet":
    parser.add_argument('--data_root', default='/path/to/ILSVRC2012/', help='path of data root')
    parser.add_argument('--test_list', default='/val.txt', help='path of test set list file')
    parser.add_argument('--train_list', default='/train.txt', help='path of train set list file')
parser.add_argument('--start_epoch', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate for optimizers.')


if parser.parse_args().generator_mode == "math":
    parser.add_argument('--save_folder', default=Save_path+parser.parse_args().dataset+'/detector/math/'+parser.parse_args().back_bone+"/"
                        +parser.parse_args().mask_shape+"/", help='Directory for saving checkpoint models')
else:
    parser.add_argument('--save_folder', default=Save_path+parser.parse_args().dataset+'/detector/'+parser.parse_args().back_bone+"/"
                        +parser.parse_args().mask_shape+"/", help='Directory for saving checkpoint models')
parser.add_argument('--log_folder', default='logs/'+'/'+parser.parse_args().back_bone+"/"+parser.parse_args().mask_shape+"/",
                    help='Directory for saving log of tensorboard')
parser.add_argument('--temp_folder', default='temp/'+'/'+parser.parse_args().back_bone+"/"+parser.parse_args().mask_shape+"/",
                    help='Directory for saving debug image')
args = parser.parse_args()
print("backbone:", args.back_bone,  "Mask type:", args.mask_shape)
print("weight path:", args.save_folder, "log path:", args.log_folder, "debug path:", args.temp_folder)
print("dataset:", args.dataset, "generator mode:", args.generator_mode)


if args.dataset == "CUB":
    saveEpoch = 20
    endEpoch = 30
elif args.dataset == "ImageNet":
    saveEpoch = 1
    endEpoch = 200

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.log_folder):
    os.mkdir(args.log_folder)
if not os.path.exists(args.temp_folder):
    os.mkdir(args.temp_folder)

def adjust_lr(epoch, start_epoch, end_epoch):
    if epoch > start_epoch:
        lr_rate = max(0, (end_epoch - epoch) * 1.0) / (end_epoch - start_epoch)
    else:
        lr_rate = 1.0
    return lr_rate

logwriter = SummaryWriter(args.log_folder)


GCNet = GC_NetModule(args.dataset, args.nz, args.nc, args.ngf, args.class_backbone, args.Classifier,
                     args.back_bone, args.mask_shape, args.generator_mode, args.generator)
Detector = GCNet.getDetector()
Classifier = GCNet.getClassifier()
Generator = GCNet.getGenerator()

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    Detector = torch.nn.DataParallel(Detector).cuda()
    Classifier = torch.nn.DataParallel(Classifier).cuda()
    Generator = torch.nn.DataParallel(Generator).cuda()


transform = GCNet.getTransformer(args.debug)
if args.debug == True:
    trainData = ImageList(root=args.data_root,
                  fileList=args.train_list, transform=transform, train=True, debug=True, shuffle=True, dataset=args.dataset)
else:
    trainData = ImageList(root=args.data_root,
                      fileList=args.train_list, transform=transform, train=True, shuffle=True, dataset=args.dataset)


trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.workers, pin_memory=True)
optimizer = GCNet.getOptim(args.lr)

cost_fun = tnn.CrossEntropyLoss()
activte_fun = tnn.Sigmoid()
softmax_fun = tnn.Softmax(dim=-1)
resize_fun = tnn.Upsample([299, 299])

# Train the model
counter = 0
total_loss = 0
total_arealoss = 0
total_objloss = 0
total_backloss = 0
Detector.train()
print("each epoch:", len(trainLoader), "iteration")
last_time = time.time()
for epoch in range(args.start_epoch, endEpoch):
    for trainData in trainLoader:
        if args.debug == True:
            if args.dataset == "CUB":
                images, visiImg, labels, bboxes = trainData
                visiImg = visiImg.cuda()
            else:
                images, visiImg, labels = trainData
                visiImg = visiImg.cuda()
        else:
            images, labels = trainData
        images = images.cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = Detector(images)
        LocSig = activte_fun(outputs)

        if args.generator_mode == "math":
            if args.mask_shape == "rectangle":
                MaskOut = Generator(LocSig[:, 0].view(LocSig.shape[0], 1), LocSig[:, 1].view(LocSig.shape[0], 1),
                                     LocSig[:, 2].view(LocSig.shape[0], 1), LocSig[:, 3].view(LocSig.shape[0], 1))
            if args.mask_shape == "rotaEllipse":
                MaskOut = Generator(LocSig[:, 0].view(LocSig.shape[0], 1), LocSig[:, 1].view(LocSig.shape[0], 1),
                                     LocSig[:, 2].view(LocSig.shape[0], 1)*0.5, LocSig[:, 3].view(LocSig.shape[0], 1)*0.5,
                                     LocSig[:, 4].view(LocSig.shape[0], 1) - 0.5)
            elif args.mask_shape == "rotaRectangle":
                MaskOut = Generator(LocSig[:, 0].view(LocSig.shape[0], 1), LocSig[:, 1].view(LocSig.shape[0], 1),
                                     LocSig[:, 2].view(LocSig.shape[0], 1), LocSig[:, 3].view(LocSig.shape[0], 1),
                                     LocSig[:, 4].view(LocSig.shape[0], 1) - 0.5)
        else:
            if args.mask_shape == "rectangle":
                MaskIn = torch.cat([LocSig[:, :2] - LocSig[:, 2:], LocSig[:, :2] + LocSig[:, 2:]], 1)
            elif args.mask_shape == "rotaRectangle" or args.mask_shape == "rotaEllipse":
                MaskIn = torch.stack([LocSig[:, 0], LocSig[:, 1], LocSig[:, 2], LocSig[:, 3], LocSig[:, 4]-0.5], 1)

            if args.class_backbone == "inceptionV3":
                MaskOut_224 = Generator(MaskIn)
                MaskOut = resize_fun(MaskOut_224)
            else:
                MaskOut = Generator(MaskIn)
            MaskOut = activte_fun(MaskOut)

        MaskOut_3 = torch.cat([MaskOut, MaskOut], 1)
        MaskOut_3 = torch.cat([MaskOut_3, MaskOut], 1)
        ReMaskOut_3 = 1 - MaskOut_3

        #object images
        ClassifiOut = Classifier(images*MaskOut_3)
        #background images
        ReClassOut = Classifier(images*ReMaskOut_3)

        #loss function
        area_loss = torch.mean(LocSig[:, 2] * LocSig[:, 3])
        ReClassOut = softmax_fun(ReClassOut)
        obj_loss = cost_fun(ClassifiOut, labels)
        eps = torch.finfo(ReClassOut.dtype).eps
        entropy = torch.sum(ReClassOut * torch.log(1 / (ReClassOut+eps)))/ReClassOut.shape[0]
        back_loss = 5.3 - entropy
        if args.generator_mode == "math":
            loss = obj_loss + back_loss + area_loss*2.5
        else:
            if args.loss == "Obj":
                loss = obj_loss
            elif args.loss == "ObjBack":
                loss = obj_loss + back_loss
            elif args.loss == "areaObj":
                loss = obj_loss + area_loss
            elif args.loss == "areaObjBack":
                loss = obj_loss + back_loss + area_loss
        #Parameter update
        loss.backward()
        optimizer.step()

        #loss visi
        total_loss += loss.item()
        total_objloss += obj_loss.item()
        total_backloss += back_loss.item()
        total_arealoss += area_loss.item()

        counter = counter + 1
        if counter % 100 == 0:
            print("100 iteration spend:", format((time.time()-last_time)/1000, '.3f'), 's')
            last_time = time.time()
            GCNet.loss_debug(total_loss, total_objloss, total_backloss, total_arealoss, counter, logwriter)
            if args.debug == True:
                GCNet.img_debug(visiImg, MaskOut_3, args.temp_folder)

            total_loss = 0
            total_objloss = 0
            total_backloss = 0
            total_arealoss = 0

    epoch_rate = adjust_lr(epoch, 20, endEpoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * epoch_rate

    if epoch % saveEpoch == 0:
        print("Epoch:%d", epoch)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            torch.save(Detector.module.state_dict(), args.save_folder + '/detector_' + str(epoch) + '.pkl')
        else:
            torch.save(Detector.state_dict(), args.save_folder + '/detector_' + str(epoch) + '.pkl')
            print('save ', args.save_folder + '/detector_' + str(epoch) + '.pkl')
print("Epoch:%d", epoch)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    torch.save(Detector.module.state_dict(), args.save_folder + '/detector_' + str(epoch) + '.pkl')
else:
    torch.save(Detector.state_dict(), args.save_folder + '/detector_' + str(epoch) + '.pkl')
print('save ', args.save_folder + '/detector_' + str(epoch) + '.pkl')