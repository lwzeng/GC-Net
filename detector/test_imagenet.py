import sys
sys.path.append("..")
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import os
import argparse
import numpy as np
from img_loader import ImageList
from PIL import Image, ImageDraw
from PIL import ImageFont
from module import *
from generator.modul import *
import cv2


np.set_printoptions(precision=3, suppress=True)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 'Obj, areaObj, ObjBack, areaObjBack'
loss_type = "areaObjBack"
Save_path = "/path/to/model/"

parser = argparse.ArgumentParser(description='classifier for loss')
parser.add_argument('--workers', default=1, type=int, help='worker number')
parser.add_argument('--dataset', default="ImageNet", help='CUB, ImageNet')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--back_bone', default="inceptionV3", help='backbone type:vgg16, resnet50, googleNet,inceptionV3')
parser.add_argument('--generator_mode', default="CNN", help='math, CNN')
parser.add_argument('--class_backbone', default="inceptionV3", help='class backbone:vgg16, resnet50, googleNet,inceptionV3')
parser.add_argument('--mask_shape', default="rotaRectangle", help='mask type:rectangle, rotaRectangle, quadrangle, rotaEllipse')
parser.add_argument('--nc', default=1, type=int, help='Number of channels in the output images.')
parser.add_argument('--nz', default=1, type=int, help='Number of channels in the input images.')
parser.add_argument('--ngf', default=32, type=int, help='Size of feature maps in generator.')
parser.add_argument('--debug', default=False, type=bool, help='debug or not')
parser.add_argument('--detector', default="/NAS_REMOTE/weizeng/model/weakdetection/ImageNet/detector/"+parser.parse_args().back_bone+"/"+parser.parse_args().mask_shape+"/single/detector_29.pkl", help='backbone type')
parser.add_argument('--Classifier', default=Save_path+"CUB/classifier/"+parser.parse_args().class_backbone+"/single/cub200_195.pkl", help='backbone type')

if parser.parse_args().mask_shape == "rectangle":
    parser.add_argument('--generator', default=Save_path+"/coord2mask/"+parser.parse_args().mask_shape+"/coord2map_110000.pth", help='backbone type')
else:
    parser.add_argument('--generator', default=Save_path+"/coord2mask/"+parser.parse_args().mask_shape+"/45degree/coord2map_119999.pth", help='backbone type')

parser.add_argument('--data_root', default='/home/weizeng/ILSVRC2012/', help='path of data root')
parser.add_argument('--test_list', default='/val.txt', help='path of test set list file')
parser.add_argument('--train_list', default='/train.txt', help='path of train set list file')

args = parser.parse_args()

print(args.detector)
# mean_vals = [0.485, 0.456, 0.406]
# std_vals = [0.229, 0.224, 0.225]

if args.back_bone == "inceptionV3":
    input_size = 299
else:
    input_size = 224
if args.back_bone == "inceptionV3":
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def my_collate(batch):
    images = []
    visiImg = []
    labels = []
    bboxes = []
    for sample in batch:
        images.append(sample[0])
        visiImg.append(sample[1])
        labels.append(torch.tensor(sample[2]))
        bboxes.append(torch.FloatTensor(sample[3]))
    return torch.stack(images, 0), torch.stack(visiImg, 0), torch.stack(labels, 0), bboxes

testData = ImageList(root=args.data_root,
                      fileList=args.test_list, transform=transform, val=True, debug=True, shuffle=False, dataset="ImageNet")

testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate, num_workers=16)

evaluate_tool = evaluate_tool(args.mask_shape, loss_type)
GCNet = GC_NetModule(args.dataset, args.nz, args.nc, args.ngf, args.class_backbone, args.Classifier,
                     args.back_bone, args.mask_shape, args.generator_mode, args.generator)
Detector = GCNet.getDetector()
Detector.load_state_dict(torch.load(args.detector))
Classifier = GCNet.getClassifier()
Generator = GCNet.getGenerator()

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    Detector = torch.nn.DataParallel(Detector).cuda()
    Classifier = torch.nn.DataParallel(Classifier).cuda()

activte_fun = tnn.Sigmoid()
resize_fun = tnn.Upsample([299, 299])

# Train the model
counter = 0
Detector.eval()
Classifier.eval()
correct_1 = 0
correct_5 = 0
Lcorrect_1 = 0
Lcorrect_5 = 0
Corcorrect = 0
total = 0

for images, visiImg, labels, bboxes in testLoader:
    images = images.cuda()
    labels = Variable(labels).cuda()
    outputs = Detector(images)

    LocSig = activte_fun(outputs)
    if args.debug == True:
        if args.mask_shape == "rotaRectangle" or args.mask_shape == "rotaEllipse":
            MaskIn = torch.stack([LocSig[:, 0], LocSig[:, 1], LocSig[:, 2], LocSig[:, 3], LocSig[:, 4] - 0.5], 1)
            if args.class_backbone == "inceptionV3":
                MaskOut_224 = Generator(MaskIn)
                MaskOut = resize_fun(MaskOut_224)
            MaskOut = activte_fun(MaskOut)
            MaskOut = MaskOut.cpu().detach().numpy()
        elif args.mask_shape == "rectangle":
            MaskIn = torch.cat([LocSig[:, :2] - LocSig[:, 2:], LocSig[:, :2] + LocSig[:, 2:]], 1)
            if args.class_backbone == "inceptionV3":
                MaskOut_224 = Generator(MaskIn)
                MaskOut = resize_fun(MaskOut_224)
            MaskOut = activte_fun(MaskOut)
            MaskOut = MaskOut.cpu().detach().numpy()

    # print(MaskOut.shape)
    LocSig = activte_fun(outputs)
    ClassOut = Classifier(images)
    confiOut = ClassOut
    LocSig = LocSig.cpu().detach().numpy()
    if args.mask_shape == "rotaRectangle":
        LocSig[:, 4] = LocSig[:, 4] - 0.5
        OutLoc = evaluate_tool.rotaPoint(LocSig)
    elif args.mask_shape == "rotaEllipse":
        LocSig[:, 4] = LocSig[:, 4] - 0.5
        OutLoc = LocSig
    _, pred = confiOut.topk(5, 1, True, True)
    pred = pred.cpu()
    pred = pred.t()
    saveImages = visiImg.detach().numpy()

    predi_boxes = torch.zeros([LocSig.shape[0], 4])
    if args.mask_shape == "rectangle":
        predi_boxes[:, 2:] = torch.tensor(LocSig[:, :2] + LocSig[:, 2:])*input_size
        predi_boxes[:, :2] = torch.tensor(LocSig[:, :2] - LocSig[:, 2:])*input_size
    elif args.mask_shape == "rotaRectangle":
        for img_index in range(saveImages.shape[0]):
            #rotaRectangle
            predi_boxes[img_index, 0] = int(min(OutLoc[img_index * 4:img_index * 4 + 4, 0]) * input_size)
            predi_boxes[img_index, 1] = int(min(OutLoc[img_index * 4:img_index * 4 + 4, 1]) * input_size)
            predi_boxes[img_index, 2] = int(max(OutLoc[img_index * 4:img_index * 4 + 4, 0]) * input_size)
            predi_boxes[img_index, 3] = int(max(OutLoc[img_index * 4:img_index * 4 + 4, 1]) * input_size)
    elif args.mask_shape == "rotaEllipse":
        for img_index in range(saveImages.shape[0]):
            gt = np.zeros([input_size, input_size], dtype=np.uint8)
            gt = cv2.ellipse(gt, (int(OutLoc[img_index, 0]*input_size), int(OutLoc[img_index, 1]*input_size)),
                (int(OutLoc[img_index, 2]*input_size*0.5), int(OutLoc[img_index, 3]*input_size*0.5)), 180*OutLoc[img_index, 4], 0, 360, 255, -1)
            gt_index = np.where(gt > 125)
            gt_index = np.array(gt_index)
            gt_index = gt_index.transpose(1, 0)
            gt_index = gt_index[:, [1, 0]]
            box = cv2.boundingRect(gt_index)
            predi_boxes[img_index, 0] = int(box[0])
            predi_boxes[img_index, 1] = int(box[1])
            predi_boxes[img_index, 2] = int(box[0] + box[2])
            predi_boxes[img_index, 3] = int(box[1] + box[3])

    IOU = evaluate_tool.IOUFunciton(predi_boxes.numpy(), bboxes)
    counter = counter + 1
    if args.debug == True:
        evaluate_tool.debugOutReult(saveImages, MaskOut, labels, LocSig, OutLoc, bboxes, counter)

    IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
    IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))
    total += labels.size(0)
    correct = pred.eq(labels.cpu().view(1, -1).expand_as(pred))
    temp_1 = correct[:1, :].view(-1) * IOU.byte()
    temp_5 = torch.sum(correct[:5, :], 0).view(-1).byte() * IOU.byte()
    C_correct_1 = correct[:1, :].view(-1).float().sum(0, keepdim=True)
    C_correct_5 = correct[:5, :].view(-1).float().sum(0, keepdim=True)
    correct_1 += C_correct_1
    correct_5 += C_correct_5
    Lcorrect_1 += temp_1.sum()
    Lcorrect_5 += temp_5.sum()
    Corcorrect += IOU.sum()
print('top1 val Accuracy:', (correct_1 / float(total)))
print('top5 val Accuracy:', (correct_5 / float(total)))
print('L top1 val Accuracy:', 1 - (Lcorrect_1.item() / float(total)))
print('L top5 val Accuracy:', 1 - (Lcorrect_5.item() / float(total)))
print("core Accuracy:", Corcorrect / total)



