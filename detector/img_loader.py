import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def default_loader(root_path, path, dataset):
    if dataset == "CUB":
        path = root_path + "/images/" + path
    elif dataset == "ImageNet":
        path = root_path + "/" + path
    # with open(path, 'rb') as f:
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

def bboxes_reader(path, dataset):
    if dataset == "ImageNet":
        bboxes_list = {}
        bboxes_file = open(path + "/val.txt")
        for line in bboxes_file:
            line = line.split('\n')[0]
            line = line.split(' ')[0]
            labelIndex = line
            line = line.split("/")[-1]
            line = line.split(".")[0]+".xml"
            bbox_path = "/data/weizeng/dataset/ILSVRC2012/val_boxes/val/" + line
            tree = ET.ElementTree(file=bbox_path)
            root = tree.getroot()
            ObjectSet = root.findall('object')
            bbox_line = []
            for Object in ObjectSet:
                BndBox = Object.find('bndbox')
                xmin = BndBox.find('xmin').text
                ymin = BndBox.find('ymin').text
                xmax = BndBox.find('xmax').text
                ymax = BndBox.find('ymax').text
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                bbox_line.append([xmin, ymin, xmax, ymax])
            bboxes_list[labelIndex] = bbox_line
        return bboxes_list

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


# MingYan modify
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
        debug=False,
        dataset="CUB"):
        self.root = root
        self.val = val
        self.train = train
        self.mask = mask
        self.suffle = shuffle
        self.reserve = reserve
        self.dataset = dataset
        self.debug = debug
        self.imagenet_size = 299
        if self.val:
            self.imgList = list_reader(self.root + fileList)
        elif self.train:
            self.imgList = list_reader(self.root + fileList)

        # if self.mask == True:
        self.bboxes = bboxes_reader(self.root, self.dataset)
        if self.suffle == True:
           random.shuffle(self.imgList)
        self.transform = transform
        self.transform_1 = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        self.loader = loader


    def __getitem__(self, index):

        img_name, label = self.imgList[index]
        img = self.loader(self.root, img_name, self.dataset)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            pass
        if self.mask == True:
            mask = np.zeros([img.size[1], img.size[0]], dtype=np.float)
            xmin , ymin, width, height = self.bboxes[img_name]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + width)
            ymax = int(ymin + height)
            bboxes = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
            mask = cv2.drawContours(mask, [bboxes], -1, 1, -1)

            if self.reserve == True:
                mask = 1 - mask
            img = np.array(img)
            for i in range(img.shape[2]):
                img[:, :, i] = img[:, :, i] * mask
            img = Image.fromarray(img)
        if self.debug == False:
            img = self.transform(img)
            return img, label
        else:
            if self.dataset == "CUB":
                bbox = self.bboxes[img_name]
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                bbox[0] = bbox[0] * (256/img.size[0]) - 16
                bbox[1] = bbox[1] * (256/img.size[1]) - 16
                bbox[2] = bbox[2] * (256/img.size[0]) - 16
                bbox[3] = bbox[3] * (256/img.size[1]) - 16
                bbox = np.array(bbox)
            elif self.dataset == "ImageNet":
                if self.train == True:
                    visimg = self.transform(img)
                    img = self.transform_1(visimg)
                    return img, visimg, label
                else:
                    bboxes = self.bboxes[img_name]
                    newBboxes = []
                    for bbox_i in range(len(bboxes)):
                        bbox = bboxes[bbox_i]
                        bbox[0] = bbox[0] * (self.imagenet_size / img.size[0])
                        bbox[1] = bbox[1] * (self.imagenet_size / img.size[1])
                        bbox[2] = bbox[2] * (self.imagenet_size / img.size[0])
                        bbox[3] = bbox[3] * (self.imagenet_size / img.size[1])
                        newBboxes.append(bbox)
            visimg = self.transform(img)
            img = self.transform_1(visimg)
            if self.dataset == "CUB":
                return img, visimg, label, bbox
            else:
                return img, visimg, label, newBboxes
    def __len__(self):
        return len(self.imgList)
