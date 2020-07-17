import sys
sys.path.append("..")
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import argparse
from img_loader import ImageList
from generator.generator_math import *
from module import *
import cv2

# 'Obj, areaObj, ObjBack, areaObjBack'
loss_type = "areaObjBack"

np.set_printoptions(precision=3, suppress=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

Save_path = "/NAS_REMOTE/weizeng/model/weakdetection/"

parser = argparse.ArgumentParser(description='classifier for loss')
parser.add_argument('--workers', default=1, type=int, help='worker number')
parser.add_argument('--dataset', default="CUB", help='CUB, ImageNet')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--back_bone', default="vgg16", help='backbone type:vgg16, resnet50, googleNet, inceptionV3')
parser.add_argument('--generator_mode', default="CNN", help='math, CNN')
parser.add_argument('--class_backbone', default="vgg16", help='backbone type:vgg16, resnet50, googleNet, inceptionV3')
parser.add_argument('--mask_shape', default="rotaEllipse", help='mask type:rectangle, rotaRectangle, quadrangle, rotaEllipse')
parser.add_argument('--nc', default=1, type=int, help='Number of channels in the output images.')
parser.add_argument('--nz', default=1, type=int, help='Number of channels in the input images.')
parser.add_argument('--ngf', default=32, type=int, help='Size of feature maps in generator.')

parser.add_argument('--debug', default=False, type=bool, help='debug or not')
if parser.parse_args().generator_mode == "math":
    parser.add_argument('--detector', default=Save_path+"CUB/detector/math/"+parser.parse_args().back_bone+"/"+parser.parse_args().mask_shape+"/smooth"+"/detector_199.pkl", help='backbone type')
else:
    parser.add_argument('--detector', default=Save_path+"CUB/detector/"+parser.parse_args().back_bone+"/"+parser.parse_args().mask_shape+"/"+"detector_199.pkl", help='backbone type')

parser.add_argument('--Classifier', default=Save_path+"CUB/classifier/"+parser.parse_args().class_backbone+"/single/cub200_199.pkl", help='backbone type')
if parser.parse_args().mask_shape == "rectangle":
    parser.add_argument('--generator', default=Save_path+"/coord2mask/"+parser.parse_args().mask_shape+"/coord2map_110000.pth", help='backbone type')
else:
    parser.add_argument('--generator', default=Save_path+"/coord2mask/"+parser.parse_args().mask_shape+"/45degree/coord2map_119999.pth", help='backbone type')
parser.add_argument('--data_root', default='/data/weizeng/code/DANet/data/CUB_200_2011/', help='path of data root')
parser.add_argument('--test_list', default='list/test.txt', help='path of test set list file')
parser.add_argument('--train_list', default='list/train.txt', help='path of train set list file')
args = parser.parse_args()
print(args.detector)

mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
])


evaluate_tool = evaluate_tool(args.mask_shape, loss_type)
GCNet = GC_NetModule(args.dataset, args.nz, args.nc, args.ngf, args.class_backbone, args.Classifier,
                     args.back_bone, args.mask_shape, args.generator_mode, args.generator)
Detector = GCNet.getDetector()
Detector.load_state_dict(torch.load(args.detector))
Classifier = GCNet.getClassifier()
Generator = GCNet.getGenerator()


testData = ImageList(root=args.data_root,
                      fileList=args.test_list, transform=transform, val=True, debug=True)

testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False, num_workers=16)


if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    Detector = torch.nn.DataParallel(Detector).cuda()
    Classifier = torch.nn.DataParallel(Classifier).cuda()

activte_fun = tnn.Sigmoid()
counter = 0
Detector.eval()
Classifier.eval()
correct_1 = 0
correct_5 = 0
Lcorrect_1 = 0
Lcorrect_5 = 0
Corcorrect = 0
prect_size = np.zeros([200, 2])
gt_size = np.zeros([200, 2])
eachError = np.zeros([200, 2])
total = 0


for images, visiImg, labels, bboxes in testLoader:
    images = images.cuda()
    labels = Variable(labels).cuda()
    outputs = Detector(images)
    LocSig = activte_fun(outputs)
    if args.debug == True:
        if args.generator_mode == "math":
            if args.mask_shape == "rotaEllipse":
                MaskOut = Generator(LocSig[:, 0].view(LocSig.shape[0], 1), LocSig[:, 1].view(LocSig.shape[0], 1),
                                     LocSig[:, 2].view(LocSig.shape[0], 1) * 0.5,
                                     LocSig[:, 3].view(LocSig.shape[0], 1) * 0.5,
                                     LocSig[:, 4].view(LocSig.shape[0], 1) - 0.5)
            elif args.mask_shape == "rotaRectangle":
                MaskOut = Generator(LocSig[:, 0].view(LocSig.shape[0], 1), LocSig[:, 1].view(LocSig.shape[0], 1),
                                     LocSig[:, 2].view(LocSig.shape[0], 1), LocSig[:, 3].view(LocSig.shape[0], 1),
                                     LocSig[:, 4].view(LocSig.shape[0], 1) - 0.5)
                MaskOut = MaskOut.cpu().detach().numpy()
        else:
            if args.mask_shape == "rotaRectangle" or args.mask_shape == "rotaEllipse":
                MaskIn = torch.stack([LocSig[:, 0], LocSig[:, 1], LocSig[:, 2], LocSig[:, 3], LocSig[:, 4] - 0.5], 1)
                MaskOut = Generator(MaskIn)
                MaskOut = activte_fun(MaskOut)
                MaskOut = MaskOut.cpu().detach().numpy()
            elif args.mask_shape == "rectangle":
                MaskIn = torch.cat([LocSig[:, :2] - LocSig[:, 2:], LocSig[:, :2] + LocSig[:, 2:]], 1)
                MaskOut = Generator(MaskIn)
                MaskOut = activte_fun(MaskOut)
                MaskOut = MaskOut.cpu().detach().numpy()
    # print(MaskOut.shape)
    LocSig = activte_fun(outputs)
    ClassOut = Classifier(images)
    confiOut = ClassOut
    LocSig = LocSig.cpu().detach().numpy()

    if args.mask_shape == "rectangle":
        if args.generator_mode == "math":
            LocSig[:, 2:] = LocSig[:, 2:]*0.5
    if args.mask_shape == "rotaRectangle":
        if args.generator_mode == "math":
            LocSig[:, 4] = 0.5 - LocSig[:, 4]
        else:
            LocSig[:, 4] = LocSig[:, 4] - 0.5
        OutLoc = evaluate_tool.rotaPoint(LocSig)
    elif args.mask_shape == "rotaEllipse":
        LocSig[:, 4] = LocSig[:, 4] - 0.5
        OutLoc = LocSig
    _, pred = confiOut.topk(5, 1, True, True)
    pred = pred.cpu()
    pred = pred.t()

    saveImages = visiImg.detach().numpy()
    counter = counter + 1
    if args.debug == True:
        evaluate_tool.debugOutReult(saveImages, MaskOut, labels, LocSig, OutLoc, bboxes, counter)

    predi_boxes = torch.zeros([LocSig.shape[0], 4])
    if args.mask_shape == "rectangle":
        predi_boxes[:, 2:] = torch.tensor(LocSig[:, :2] + LocSig[:, 2:])*224
        predi_boxes[:, :2] = torch.tensor(LocSig[:, :2] - LocSig[:, 2:])*224
    elif args.mask_shape == "rotaRectangle":
        for img_index in range(saveImages.shape[0]):
            predi_boxes[img_index, 0] = int(min(OutLoc[img_index * 4:img_index * 4 + 4, 0]) * 224)
            predi_boxes[img_index, 1] = int(min(OutLoc[img_index * 4:img_index * 4 + 4, 1]) * 224)
            predi_boxes[img_index, 2] = int(max(OutLoc[img_index * 4:img_index * 4 + 4, 0]) * 224)
            predi_boxes[img_index, 3] = int(max(OutLoc[img_index * 4:img_index * 4 + 4, 1]) * 224)
    elif args.mask_shape == "rotaEllipse":
        for img_index in range(saveImages.shape[0]):
            gt = np.zeros([224, 224], dtype=np.uint8)
            gt = cv2.ellipse(gt, (int(OutLoc[img_index, 0]*224), int(OutLoc[img_index, 1]*224)),
                (int(OutLoc[img_index, 2]*224*0.5), int(OutLoc[img_index, 3]*224*0.5)), 180*OutLoc[img_index, 4], 0, 360, 255, -1)
            gt_index = np.where(gt > 125)
            gt_index = np.array(gt_index)
            gt_index = gt_index.transpose(1, 0)
            gt_index = gt_index[:, [1, 0]]
            box = cv2.boundingRect(gt_index)
            predi_boxes[img_index, 0] = int(box[0])
            predi_boxes[img_index, 1] = int(box[1])
            predi_boxes[img_index, 2] = int(box[0] + box[2])
            predi_boxes[img_index, 3] = int(box[1] + box[3])

    predi_boxes = predi_boxes.data.float()
    bboxes = bboxes.data.cpu().float()
    inter = evaluate_tool.intersect(predi_boxes, bboxes)
    area_a = (predi_boxes[:, 2]-predi_boxes[:, 0]) * (predi_boxes[:, 3]-predi_boxes[:, 1])
    area_b = (bboxes[:, 2]-bboxes[:, 0]) * (bboxes[:, 3]-bboxes[:, 1])
    union = area_a + area_b - inter
    IOU = inter / union
    IOU = torch.where(IOU <= 0.5, IOU, torch.ones(IOU.shape[0]))
    IOU = torch.where(IOU > 0.5, IOU, torch.zeros(IOU.shape[0]))

    total += labels.size(0)
    correct = pred.eq(labels.cpu().view(1, -1).expand_as(pred))
    temp_1 = correct[:1, :].view(-1) * IOU.byte()
    temp_5 = torch.sum(correct[:5, :], 0).view(-1).byte() * IOU.byte()
    correct_1 += correct[:1, :].view(-1).float().sum(0, keepdim=True)
    correct_5 += correct[:5, :].view(-1).float().sum(0, keepdim=True)
    Lcorrect_1 += temp_1.sum()
    Lcorrect_5 += temp_5.sum()
    Corcorrect += IOU.sum()
print(Lcorrect_1.item())
print(Lcorrect_5.item())
print('top1 val Accuracy:', (correct_1 / float(total)))
print('top5 val Accuracy:', (correct_5 / float(total)))
print('L top1 val Accuracy:', (1-Lcorrect_1.item() / float(total))*100)
print('L top5 val Accuracy:', (1-Lcorrect_5.item() / float(total))*100)
print("core Accuracy:", Corcorrect / total)



