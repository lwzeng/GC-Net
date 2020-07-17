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
from torchvision import datasets, models, transforms
from PIL import Image, ImageDraw
import cv2
import time

# np.set_printoptions(precision=3, suppress=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

Save_path = "/NAS_REMOTE/weizeng/model/weakdetection/"

parser = argparse.ArgumentParser(description='classifier for loss')
parser.add_argument('--workers', default=16, type=int, help='worker number')
parser.add_argument('--batch_size', default=10, type=int, help='batch size')
parser.add_argument('--dataset', default="CUB", help='CUB, ImageNet')
parser.add_argument('--ten_crop', default=True, help='ten crop or not')
parser.add_argument('--class_backbone', default="vgg16", help='backbone type:vgg16, googleNet, resnet50')
if parser.parse_args().dataset == "CUB":
    parser.add_argument('--Classifier', default=Save_path+"CUB/classifier/"+parser.parse_args().class_backbone+"/single/cub200_199.pkl", help='backbone type')
    parser.add_argument('--data_root', default='/data/weizeng/code/DANet/data/CUB_200_2011/', help='path of data root')
    parser.add_argument('--test_list', default='list/test.txt', help='path of test set list file')
    parser.add_argument('--train_list', default='list/train.txt', help='path of train set list file')
elif parser.parse_args().dataset == "ImageNet":
    parser.add_argument('--data_root', default='/data/weizeng/dataset/ILSVRC2012/', help='path of data root')
    parser.add_argument('--test_list', default='/val.txt', help='path of test set list file')
    parser.add_argument('--train_list', default='/train.txt', help='path of train set list file')
args = parser.parse_args()

print(args.dataset, args.class_backbone)

mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean_vals, std=std_vals)

if args.ten_crop == True:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop((224, 224)),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize(mean=mean_vals, std=std_vals)(crop) for crop in crops])),
    ])
else:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals,
                             std=std_vals),
    ])

testData = ImageList(root=args.data_root, mask=False, reserve=False,
                      fileList=args.test_list, transform=transform, val=True, dataset=args.dataset)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False, num_workers=16)

if args.class_backbone == "resnet50":
    Classifier = models.resnet50(pretrained=True)
    if args.dataset == "CUB":
        Classifier.fc = tnn.Linear(in_features=2048, out_features=200, bias=True)
elif args.class_backbone == "resnet18":
    Classifier = models.resnet18(pretrained=True)
    if args.dataset == "CUB":
        Classifier.fc = tnn.Linear(in_features=2048, out_features=200, bias=True)
elif args.class_backbone == "vgg16":
    Classifier = models.vgg16(pretrained=True)
    if args.dataset == "CUB":
        Classifier.classifier[6] = tnn.Linear(in_features=4096, out_features=200, bias=True)
elif args.class_backbone == "googleNet":
    # Classifier = models.inception_v3(pretrained=True, transform_input=False)
    Classifier = models.googlenet(pretrained=True, transform_input=False)
    if args.dataset == "CUB":
        # Classifier.fc = tnn.Linear(in_features=2048, out_features=200, bias=True)
        Classifier.fc = tnn.Linear(in_features=1024, out_features=200, bias=True)

Classifier.cuda()
if args.dataset == "CUB":
    Classifier.load_state_dict(torch.load(args.Classifier))

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    Classifier = torch.nn.DataParallel(Classifier).cuda()

print(args.Classifier)


Classifier.eval()
correct_1 = 0
correct_5 = 0
total = 0
total_entropy = 0

counter = 0
print("test:", len(testLoader), "iteration")
softmax_fun = tnn.Softmax(dim=-1)
softMax = tnn.Softmax(1)
for images, labels in testLoader:
    counter = counter + 1
    images = images.cuda()
    labels = labels.cuda()
    if args.ten_crop == True:
        bs, ncrops, c, h, w = images.size()
        images = images.view(-1, c, h, w)
    ClassOut = Classifier(images)
    if args.ten_crop == True:
        ClassOut = ClassOut.view(bs, ncrops, -1).mean(1)
    confiOut = ClassOut
    _, pred = confiOut.topk(5, 1, True, True)
    pred = pred.cpu()
    pred = pred.t()
    total += labels.size(0)
    correct = pred.eq(labels.cpu().view(1, -1).expand_as(pred))
    correct_1 += correct[:1].view(-1).float().sum(0, keepdim=True)*100
    correct_5 += correct[:5].view(-1).float().sum(0, keepdim=True)*100

    outputs = softMax(ClassOut)
    outputs = outputs.detach().cpu().numpy()
    entropy = np.sum(outputs * np.log(1 /outputs))/outputs.shape[0]
    total_entropy += entropy
    counter = counter + 1

    if counter % 100 == 0:
        print("iteration:", counter)
print("total:", total)
print("entropy:", total_entropy/float(counter))
print('top1 val Accuracy:', (100 - correct_1 / float(total)))
print('top5 val Accuracy:', (100 - correct_5 / float(total)))
