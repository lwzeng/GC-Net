import torch
import torch.nn as tnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import os
from img_loader import ImageList as ImageList
from img_loader_hierarchy import ImageList as HierImageList
import argparse
from tensorboardX import SummaryWriter
from module import vggbasenetwork

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

Save_path = "/NAS_REMOTE/weizeng/model/weakdetection/"

parser = argparse.ArgumentParser(description='classifier for loss')
parser.add_argument('--hierarchy', default=False, type=bool, help='worker number')
parser.add_argument('--workers', default=1, type=int, help='worker number')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--resume', default="version_0/weights/coord2map_30000.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--data_root', default='/data/weizeng/code/DANet/data/CUB_200_2011/', help='path of data root')
parser.add_argument('--class_backbone', default="inceptionV3", help='backbone type:vgg16, resnet50, googleNet, inceptionV3')
parser.add_argument('--test_list', default='list/test.txt', help='path of test set list file')
parser.add_argument('--train_list', default='list/train.txt', help='path of train set list file')
parser.add_argument('--family_list', default='list/family_label.txt', help='path of train set list file')
parser.add_argument('--root_list', default='list/order_label.txt', help='path of train set list file')

parser.add_argument('--start_epoch', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--epoch', default=200, type=int, help='Number of training iteration.')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate for optimizers.')
if parser.parse_args().hierarchy == True:
    saveMode = "weights/googleNet/hierarchy"
else:
    saveMode = Save_path+'CUB/classifier/'+parser.parse_args().class_backbone+'/single/'
parser.add_argument('--save_folder', default=saveMode, help='Directory for saving checkpoint models')
parser.add_argument('--log_folder', default='logs/google/', help='Directory for saving log of tensorboard')
args = parser.parse_args()

lr_steps = [80, 150, 200]

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.log_folder):
    os.mkdir(args.log_folder)

logwriter = SummaryWriter(args.log_folder)

mean_vals = [0.485, 0.456, 0.406]
std_vals = [0.229, 0.224, 0.225]

if args.class_backbone == "googleNet":
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals,
                             std=std_vals),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals,
                             std=std_vals),
    ])
elif args.class_backbone == "inceptionV3":
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals,
                             std=std_vals),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals,
                             std=std_vals),
    ])


if args.hierarchy == True:
    trainData = HierImageList(root=args.data_root,
                          fileList=args.train_list, familyList=args.family_list, rootList=args.root_list, transform=train_transform, train=True, shuffle=True)
    testData = HierImageList(root=args.data_root,
                          fileList=args.test_list, transform=val_transform, val=True)
else:
    trainData = ImageList(root=args.data_root,
                          fileList=args.train_list, transform=train_transform, train=True, shuffle=True)
    testData = ImageList(root=args.data_root,
                         fileList=args.test_list, transform=val_transform, val=True)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True, num_workers=16)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False, num_workers=16)

if args.class_backbone == "googleNet":
    if args.hierarchy == True:
        NetMode = vggbasenetwork.vgg16(pretrained=True)
    else:
        NetMode = models.googlenet(pretrained=True, transform_input=False)
        NetMode.fc = tnn.Linear(1024, 200)
elif args.class_backbone == "inceptionV3":
    NetMode = models.inception_v3(pretrained=True, transform_input=False)
    NetMode.aux_logits = False
    NetMode.fc = tnn.Linear(2048, 200)
NetMode.cuda()


if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    NetMode = torch.nn.DataParallel(NetMode).cuda()

cost = tnn.CrossEntropyLoss()
weight_list = []
bias_list = []
fc_weight_list = []
fc_bias_list = []
for name, value in NetMode.named_parameters():
    # print(name)
    if 'fc' in name or "classifier_r" in name or "classifier_f" in name:
        if 'weight' in name:
            fc_weight_list.append(value)
        elif 'bias' in name:
            fc_bias_list.append(value)
    else:
        if 'weight' in name:
            weight_list.append(value)
        elif 'bias' in name:
            bias_list.append(value)

optimizer = torch.optim.SGD([{'params': weight_list, 'lr': args.lr},
                            {'params': bias_list, 'lr': args.lr * 2},
                            {'params': fc_weight_list, 'lr': args.lr * 10},
                            {'params': fc_bias_list, 'lr': args.lr * 20}],
                             momentum=0.9, weight_decay=5e-4)

# Train the model
counter = 0
total_loss = 0
total_loss_c = 0
total_loss_f = 0
total_loss_r = 0
for epoch in range(args.start_epoch, args.epoch):
    NetMode.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    print("len of the train set", len(trainLoader))
    for imagesAndLabel in trainLoader:
        if args.hierarchy == True:
            images, labels, famliyLabels, rootLabels = imagesAndLabel
            images = images.cuda()
            labels = Variable(labels).cuda()
            famliyLabels = Variable(famliyLabels).cuda()
            rootLabels = Variable(rootLabels).cuda()
        else:
            images, labels = imagesAndLabel
            # print(images.shape)
            # print(labels)
            images = images.cuda()
            labels = Variable(labels).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        if args.hierarchy == True:
            rootOut, parentOut, outputs = NetMode(images)
            loss_class = cost(outputs, labels)
            loss_family = cost(parentOut, famliyLabels)
            loss_root = cost(rootOut, rootLabels)
            loss = loss_class + loss_family + loss_root
        else:
            outputs = NetMode(images)
            loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

        total_loss += loss.item()
        if parser.parse_args().hierarchy == True:
            total_loss_c += loss_class.item()
            total_loss_f += loss_family.item()
            total_loss_r += loss_root.item()

        counter = counter + 1
        if counter % 50 == 0:
            if counter == 0:
                total_loss = total_loss
                if parser.parse_args().hierarchy == True:
                    total_loss_c = total_loss_c
                    total_loss_f = total_loss_f
                    total_loss_r = total_loss_r
            else:
                total_loss = total_loss/50
                if parser.parse_args().hierarchy == True:
                    total_loss_c = total_loss_c/50
                    total_loss_f = total_loss_f/50
                    total_loss_r = total_loss_r/50
            if parser.parse_args().hierarchy == True:
                print('total_loss:', total_loss, 'total_loss_c:', total_loss_c, 'total_loss_f:', total_loss_f, 'total_loss_r:', total_loss_r)
            else:
                print('total_loss:', total_loss)
            logwriter.add_scalar('Train/total_loss', total_loss, counter)
            if parser.parse_args().hierarchy == True:
                logwriter.add_scalar('Train/total_loss_c', total_loss_c, counter)
                logwriter.add_scalar('Train/total_loss_f', total_loss_f, counter)
                logwriter.add_scalar('Train/total_loss_r', total_loss_r, counter)
            total_loss = 0
            if parser.parse_args().hierarchy == True:
                total_loss_c = 0
                total_loss_f = 0
                total_loss_r = 0
    print('train Accuracy = %f' % (train_correct / float(train_total)))
    NetMode.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0
    for images, labels in testLoader:
        images = images.cuda()
        if args.hierarchy == True:
            rootOut, parentOut, outputs = NetMode(images)
        else:
            outputs = NetMode(images)
        total += labels.size(0)
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.cpu()
        pred = pred.t()
        correct = pred.eq(labels.cpu().view(1, -1).expand_as(pred))
        correct_1 += correct[:1].view(-1).float().sum(0, keepdim=True)
        correct_5 += correct[:5].view(-1).float().sum(0, keepdim=True)
    print('top1 val Accuracy:', (correct_1 / float(total)))
    print('top5 val Accuracy:', (correct_5 / float(total)))
    NetMode.train()

    if epoch in lr_steps:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    if epoch % 10 == 0:
        print("Epoch:%d", epoch)
        torch.save(NetMode.state_dict(), args.save_folder +
                   '/cub200_' + str(epoch) + '.pkl')
        print('save ', args.save_folder + '/cub200_' + str(epoch) + '.pkl')
print("Epoch:%d", epoch)
torch.save(NetMode.state_dict(), args.save_folder +
           '/cub200_' + str(epoch) + '.pkl')
print('save ', args.save_folder + '/cub200_' + str(epoch) + '.pkl')