import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import random
import cv2
import numpy as np


def default_loader(root_path, path, dataset):
    if dataset == "CUB":
        path = root_path + "/images/" + path
    elif dataset == "ImageNet":
        path = root_path + "/" + path
    with open(path, 'rb') as f:
        img = Image.open(path).convert('RGB')
    return img


def default_list_reader(fileList):
    imgList = []
    # classes = set()
    # print(fileList)
    with open(fileList, 'r') as file:
        for line in file.readlines():
            lineSplit = line.strip().split(' ')
            imgPath, label = lineSplit[0], lineSplit[1]
            # classes.add(label)
            imgList.append((imgPath, int(label)))
    # print(imgList, classes)
    return imgList

def bboxes_reader(path):
    bboxes_file = open(path + "/bounding_boxes.txt")
    images_map = open(path + "/images.txt")
    bboxes_list = {}
    file1_list = {}
    for line in bboxes_file:
        line = line.split('\n')[0]
        line = line.split(' ')
        file1_list[line[0]] = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
    file2_list = {}
    for line in images_map:
        line = line.split('\n')[0]
        line = line.split(' ')
        file2_list[line[0]] = line[1]
    for line in file1_list:
        bboxes_list[file2_list[line]] = file1_list[line]
        # print(line, file1_list[line], file2_list[line])
    return bboxes_list

def default_attr_reader(attrfile):
    attr = {}
    with open(attrfile, 'r') as file:
        # line 1 is the number of pic
        file.readline()
        # line 2 are attr names
        attrname = file.readline().strip().split(' ')
        # the rest are val
        for line in file.readlines():
            val = line.strip().split()
            pic_name = val[0]
            val.pop(0)
            img_attr = {}
            if pic_name in attr:
                img_attr = attr[pic_name]

                for i, name in enumerate(attrname, 0):
                    # maybe can store as str. do not use int
                    img_attr[name] = int(val[i])

                attr[pic_name] = img_attr
    return attr, attrname


class ImageList(data.Dataset):

    def __init__(
        self,
        root=None,
        fileList=None,
        transform=None,
        list_reader=default_list_reader,
        loader=default_loader,
        train=False,
        val=False,
        shuffle=False,
        mask=False,
        reserve=False,
        mask_rate=1.0,
        dataset="CUB"):
        self.root = root
        self.val = val
        self.train = train
        self.mask = mask
        self.suffle = shuffle
        self.reserve = reserve
        self.dataset = dataset
        self.mask_rate = mask_rate
        if self.val:
            self.imgList = list_reader(self.root + fileList)
        elif self.train:
            self.imgList = list_reader(self.root + fileList)

        if self.mask == True:
            self.bboxes = bboxes_reader(self.root)
        if self.suffle == True:
           random.shuffle(self.imgList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):

        img_name, label = self.imgList[index]
        img = self.loader(self.root, img_name, self.dataset)
        # print(img.size)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            pass
        # print(img.size[0], img.size[1])
        if self.mask == True:
            mask = np.zeros([img.size[1], img.size[0]], dtype=np.float)
            xmin , ymin, width, height = self.bboxes[img_name]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + width)
            ymax = int(ymin + height)
            width, height = width*self.mask_rate, height*self.mask_rate
            center_x = int((xmin+xmax)/2.0)
            center_y = int((ymin+ymax)/2.0)
            xmin, ymin = int(center_x-width*0.5), int(center_y-height*0.5)
            xmax, ymax = int(center_x+width*0.5), int(center_y+height*0.5)

            # print(xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin)
            bboxes = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
            mask = cv2.drawContours(mask, [bboxes], -1, 1, -1)

            if self.reserve == True:
                mask = 1 - mask
            img = np.array(img)
            # print(img.shape)
            for i in range(img.shape[2]):
                img[:, :, i] = img[:, :, i] * mask
            img = Image.fromarray(img)
        img.save("temp/"+str(int(self.mask_rate*10))+"_"+str(index) + ".png")
        img = self.transform(img)
        # print(img.shape)
        return img, label

    def __len__(self):
        return len(self.imgList)


#
# transform = transforms.Compose([
#     # transforms.Resize((256, 256)),
#     # transforms.RandomCrop((224, 224)),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                          std=[0.5, 0.5, 0.5]),
# ])
#
#
# test_load = ImageList(root='/data/weizeng/code/DANet/data/CUB_200_2011/',
#                       fileList='list/train.txt', transform=transform, train=True, mask=True, reserve=True)
# for i in range(100):
#     img, label = test_load.__getitem__(i)
#     # print(img.size, label)

# bboxes_reader('/data/weizeng/code/DANet/data/CUB_200_2011/')