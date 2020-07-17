import torch
from generator.generator_math import *
from generator.modul import *
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw


#angle -90 to 90
class GC_NetModule():
    def __init__(self, dataset, nz, nc, ngf, class_backbone, class_weight, detector_backbone,
                 mask_shape, generator_mode, generator_weight):
        super(GC_NetModule, self).__init__()

        self.dataset = dataset
        self.class_backbone = class_backbone
        self.detector_backbone = detector_backbone
        self.mask_shape = mask_shape
        self.generator_mode = generator_mode
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.generator = self.GeneratorFun(generator_weight)
        self.detector = self.DetectorFun()
        self.classifer = self.ClassifierFun(class_weight)
    def GeneratorFun(self, generator_weight):
        if self.generator_mode == "math":
            if self.mask_shape == "rectangle":
                generator = gene_Rectangle(img_h=224, img_w=224)
            elif self.mask_shape == "rotaEllipse":
                generator = gene_rotaEllipse(img_h=224, img_w=224)
            elif self.mask_shape == "rotaRectangle":
                generator = gene_rotaRectangle(img_h=224, img_w=224)
            generator = generator.cuda()
        else:
            if self.mask_shape == "rectangle":
                generator = rectGenerator(self.nz, self.nc, self.ngf).to("cuda")
            elif self.mask_shape == "rotaRectangle" or self.mask_shape == "rotaEllipse":
                generator = rotaGenerator(self.nz, self.nc, self.ngf).to("cuda")
            generator.load_state_dict(torch.load(generator_weight))
        return generator


    def DetectorFun(self):
        #detector
        if self.detector_backbone == "resnet50":
            detector = models.resnet50(pretrained=True)
            if self.mask_shape == "rectangle":
                detector.fc = tnn.Linear(in_features=2048, out_features=4, bias=True)
            elif self.mask_shape == "rotaRectangle" or self.mask_shape == "rotaEllipse":
                detector.fc = tnn.Linear(in_features=2048, out_features=5, bias=True)
            elif self.mask_shape == "quadrangle":
                detector.fc = tnn.Linear(in_features=2048, out_features=12, bias=True)
        elif self.detector_backbone == "vgg16":
            detector = models.vgg16(pretrained=True)
            if self.mask_shape == "rectangle":
                detector.classifier[6] = tnn.Linear(in_features=4096, out_features=4, bias=True)
            elif self.mask_shape == "rotaRectangle" or self.mask_shape == "rotaEllipse":
                detector.classifier[6] = tnn.Linear(in_features=4096, out_features=5, bias=True)
        elif self.detector_backbone == "googleNet":
            detector = models.googlenet(pretrained=True, transform_input=False)
            if self.mask_shape == "rectangle":
                detector.fc = tnn.Linear(1024, 4)
            elif self.mask_shape == "rotaRectangle" or self.mask_shape == "rotaEllipse":
                detector.fc = tnn.Linear(1024, 5)
        elif self.detector_backbone == "inceptionV3":
            detector = models.inception_v3(pretrained=True, transform_input=False)
            detector.aux_logits = False
            if self.mask_shape == "rectangle":
                detector.fc = tnn.Linear(2048, 4)
            elif self.mask_shape == "rotaRectangle" or self.mask_shape == "rotaEllipse":
                detector.fc = tnn.Linear(2048, 5)
        detector.cuda()
        return detector


    def ClassifierFun(self, class_weight):
        #classifier
        if self.class_backbone == "resnet50":
            classifer = models.resnet50(pretrained=True)
            if self.dataset == "CUB":
                classifer.fc = tnn.Linear(in_features=2048, out_features=200, bias=True)
        elif self.class_backbone == "vgg16":
            classifer = models.vgg16(pretrained=True)
            if self.dataset == "CUB":
                classifer.classifier[6] = tnn.Linear(in_features=4096, out_features=200, bias=True)
        elif self.class_backbone == "googleNet":
            classifer = models.googlenet(pretrained=True, transform_input=False)
            if self.dataset == "CUB":
                classifer.fc = tnn.Linear(1024, 200)
        elif self.class_backbone== "inceptionV3":
            classifer = models.inception_v3(pretrained=True, transform_input=False)
            classifer.aux_logits = False
            if self.dataset == "CUB":
                classifer.fc = tnn.Linear(2048, 200)
        if self.dataset== "CUB":
            classifer.load_state_dict(torch.load(class_weight))
        classifer.cuda()
        return classifer
    def getDetector(self):
        return self.detector

    def getClassifier(self):
        return self.classifer

    def getGenerator(self):
        return self.generator

    def getTransformer(self, debug):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        if debug == True:
            if self.dataset == "CUB":
                if self.class_backbone == "inceptionV3":
                    transform = transforms.Compose([
                        transforms.Resize((299, 299)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ])
            elif self.dataset == "ImageNet":
                transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            if self.dataset == "CUB":
                if self.class_backbone == "inceptionV3":
                    transform = transforms.Compose([
                        transforms.Resize((299, 299)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean_vals,
                                             std=std_vals),
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.RandomCrop((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean_vals,
                                             std=std_vals),
                    ])
            elif self.dataset == "ImageNet":
                transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_vals,
                                         std=std_vals),
                ])
        return transform

    def getOptim(self, lr):
        weight_list = []
        bias_list = []
        fc_weight_list = []
        fc_bias_list = []
        if self.detector_backbone == "vgg16":
            fcName = "classifier.6"
        elif self.detector_backbone == "googleNet" or self.detector_backbone == "inceptionV3":
            fcName = "fc"
        for name, value in self.detector.named_parameters():
            if fcName in name:
                if 'weight' in name:
                    fc_weight_list.append(value)
                elif 'bias' in name:
                    fc_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)

        if self.generator_mode == "math":
            optimizer = torch.optim.Adam([{'params': weight_list, 'lr': lr},
                                          {'params': bias_list, 'lr': lr * 2},
                                          {'params': fc_weight_list, 'lr': lr * 10},
                                          {'params': fc_bias_list, 'lr': lr * 20},
                                          {'params': self.generator.parameters(), 'lr': lr * 10}])
        else:
            optimizer = torch.optim.Adam([{'params': weight_list, 'lr': lr},
                                          {'params': bias_list, 'lr': lr * 2},
                                          {'params': fc_weight_list, 'lr': lr * 10},
                                          {'params': fc_bias_list, 'lr': lr * 20}])
        return optimizer

    def loss_debug(self, total_loss, total_objloss, total_backloss, total_arealoss, counter, logwriter):
        if counter == 0:
            total_loss = total_loss
            total_objloss = total_objloss
            total_backloss = total_backloss
            total_arealoss = total_arealoss
        else:
            total_loss = total_loss / 100
            total_objloss = total_objloss / 100
            total_backloss = total_backloss / 100
            total_arealoss = total_arealoss / 100

        print("iteration:", counter, "loss:", format(total_loss,'.3f'), "area_loss:", format(total_arealoss,'.3f'),
              "obj_loss:",  format(total_objloss,'.3f'), "back_loss:", format(total_backloss,'.3f'))
        logwriter.add_scalar('Train/Loss', total_loss, counter)
        logwriter.add_scalar('Train/area_loss', total_arealoss, counter)
        logwriter.add_scalar('Train/obj_loss', total_objloss, counter)
        logwriter.add_scalar('Train/back_loss', total_backloss, counter)
    def img_debug(self, visiImg, MaskOut_3, temp_folder):
        saveImages = visiImg * MaskOut_3
        saveImages = saveImages.cpu().detach().numpy()
        for img_index in range(min(32, saveImages.shape[0])):
            per_img = np.zeros([visiImg.shape[-1], visiImg.shape[-1], 3])
            per_img[:, :, 0] = saveImages[img_index, 0, :, :] * 255
            per_img[:, :, 1] = saveImages[img_index, 1, :, :] * 255
            per_img[:, :, 2] = saveImages[img_index, 2, :, :] * 255
            saveImg = Image.fromarray(per_img.astype(np.uint8))
            # print(temp_folder + "/" + str(img_index) + ".png")
            saveImg.save(temp_folder + "/" + str(img_index) + ".png")


class evaluate_tool():
    def __init__(self, mask_shape, loss_type):
        super(evaluate_tool, self).__init__()
        self.mask_shape = mask_shape
        self.loss_type = loss_type
    def rotaPoint(self, input):
        affine = np.zeros([input.shape[0], 3, 3])
        output = np.zeros([input.shape[0] * 4, 2])

        # input coordinate
        min_x = input[:, 0] - input[:, 2] * 0.5
        min_y = input[:, 1] - input[:, 3] * 0.5
        max_x = input[:, 0] + input[:, 2] * 0.5
        max_y = input[:, 1] + input[:, 3] * 0.5
        input_coord = np.stack([min_x, min_y], 1).transpose(1, 0)
        temp = np.stack([max_x, min_y], 1).transpose(1, 0)
        input_coord = np.concatenate([input_coord, temp], 1)
        temp = np.stack([max_x, max_y], 1).transpose(1, 0)
        input_coord = np.concatenate([input_coord, temp], 1)
        temp = np.stack([min_x, max_y], 1).transpose(1, 0)
        input_coord = np.concatenate([input_coord, temp], 1)
        input_coord = np.vstack((input_coord, np.ones(4 * input.shape[0])))

        # affine
        offset_x = input[:, 0]
        offset_y = input[:, 1]
        cos = np.cos(input[:, 4] * np.pi)
        sin = np.sin(input[:, 4] * np.pi)
        affine[:, 0, 0] = cos
        affine[:, 0, 1] = -sin
        affine[:, 0, 2] = offset_x * (1 - cos) + offset_y * sin
        affine[:, 1, 0] = sin
        affine[:, 1, 1] = cos
        affine[:, 1, 2] = offset_y * (1 - cos) - offset_x * sin
        affine[:, 2, 2] = 1

        for index in range(affine.shape[0]):
            per_coord = np.stack([input_coord[:, index], input_coord[:, affine.shape[0] + index],
                                  input_coord[:, 2 * affine.shape[0] + index],
                                  input_coord[:, 3 * affine.shape[0] + index]], 1)
            # print(per_coord.shape, affine[index, :, :].shape)
            tempOut = affine[index, :, :].dot(per_coord)[0:2, :].transpose(1, 0)
            output[4 * index:4 * index + 4, :] = tempOut
        return output

    def intersect(self, box_a, box_b):
        max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
        min_xy = torch.max(box_a[:, :2], box_b[:, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, 0] * inter[:, 1]

    def IOUFunciton(self, boxes_a, boxes_b):
        IOUList = np.zeros(len(boxes_b))
        for bbox_i in range(len(boxes_b)):
            box_a = boxes_a[bbox_i]
            area_a = (box_a[2] - box_a[0])*(box_a[3] - box_a[1])
            imgBoxes_b = boxes_b[bbox_i]
            tempIOU = 0
            for bbox_j in range(imgBoxes_b.shape[0]):
                box_b = imgBoxes_b[bbox_j]
                area_b = (box_b[2] - box_b[0])*(box_b[3] - box_b[1])
                intersect = (min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))*(min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
                abIOU = intersect/(area_a+area_b-intersect)
                if abIOU > tempIOU:
                    tempIOU = abIOU
            IOUList[bbox_i] = tempIOU
        return torch.tensor(IOUList, dtype=torch.float)

    def debugOutReult(self, saveImages, MaskOut, labels, LocSig, OutLoc, bboxes, counter):
        for img_index in range(saveImages.shape[0]):
            per_img = np.zeros([224, 224, 3])
            per_img[:, :, 0] = saveImages[img_index, 0, :, :] * 255
            per_img[:, :, 1] = saveImages[img_index, 1, :, :] * 255
            per_img[:, :, 2] = saveImages[img_index, 2, :, :] * 255
            saveImg = Image.fromarray(per_img.astype(np.uint8))
            draw = ImageDraw.Draw(saveImg)
            if self.mask_shape == "rectangle":
                mask_save = np.zeros([MaskOut.shape[2], MaskOut.shape[3], 3])
                mask_save[:, :, 0] = MaskOut[img_index, 0, :, :]
                mask_save[:, :, 1] = MaskOut[img_index, 0, :, :]
                mask_save[:, :, 2] = MaskOut[img_index, 0, :, :]

                xmin = (LocSig[img_index, 0].item() - LocSig[img_index, 2].item()) * 224
                ymin = (LocSig[img_index, 1].item() - LocSig[img_index, 3].item()) * 224
                xmax = (LocSig[img_index, 0].item() + LocSig[img_index, 2].item()) * 224
                ymax = (LocSig[img_index, 1].item() + LocSig[img_index, 3].item()) * 224

                # mask * image
                mask_img = np.zeros([MaskOut.shape[2], MaskOut.shape[3], 3])
                mask_img[:, :, 0] = saveImages[img_index, 0, :, :] * mask_save[:, :, 0] * 255
                mask_img[:, :, 1] = saveImages[img_index, 1, :, :] * mask_save[:, :, 1] * 255
                mask_img[:, :, 2] = saveImages[img_index, 2, :, :] * mask_save[:, :, 2] * 255
                mask_img = Image.fromarray(mask_img.astype(np.uint8))
                mask_img.save(
                    "temp/test/mask_" + str(labels[img_index].item()) + "_" + str(counter) + "_" + str(
                        img_index) + ".png")
            elif self.mask_shape == "rotaRectangle":
                # print(MaskOut.shape[0], MaskOut.shape[1], MaskOut.shape[2], MaskOut.shape[3])
                mask_save = np.zeros([MaskOut.shape[2], MaskOut.shape[3], 3])
                mask_save[:, :, 0] = MaskOut[img_index, 0, :, :]
                mask_save[:, :, 1] = MaskOut[img_index, 0, :, :]
                mask_save[:, :, 2] = MaskOut[img_index, 0, :, :]
                prebbox = [(int(OutLoc[img_index*4, 0]*224),   int(OutLoc[img_index*4, 1]*224)),
                           (int(OutLoc[img_index*4+1, 0]*224), int(OutLoc[img_index*4+1, 1]*224)),
                           (int(OutLoc[img_index*4+2, 0]*224), int(OutLoc[img_index*4+2, 1]*224)),
                           (int(OutLoc[img_index*4+3, 0]*224), int(OutLoc[img_index*4+3, 1]*224))]
                draw.line((prebbox[0], prebbox[1]), fill=(0, 0, 0), width=3)
                draw.line((prebbox[1], prebbox[2]), fill=(0, 0, 0), width=3)
                draw.line((prebbox[2], prebbox[3]), fill=(0, 0, 0), width=3)
                draw.line((prebbox[3], prebbox[0]), fill=(0, 0, 0), width=3)
                # draw.polygon(prebbox, outline=(255, 255, 255))

                xmin = int(min(OutLoc[img_index*4:img_index*4+4, 0])*224)
                ymin = int(min(OutLoc[img_index*4:img_index*4+4, 1])*224)
                xmax = int(max(OutLoc[img_index*4:img_index*4+4, 0])*224)
                ymax = int(max(OutLoc[img_index*4:img_index*4+4, 1])*224)

                # mask * image
                mask_img = np.zeros([MaskOut.shape[2], MaskOut.shape[3], 3])
                mask_img[:, :, 0] = saveImages[img_index, 0, :, :] * mask_save[:, :, 0] * 255
                mask_img[:, :, 1] = saveImages[img_index, 1, :, :] * mask_save[:, :, 1] * 255
                mask_img[:, :, 2] = saveImages[img_index, 2, :, :] * mask_save[:, :, 2] * 255
                mask_img = Image.fromarray(mask_img.astype(np.uint8))
                mask_img.save(
                    "temp/test/mask_" + str(labels[img_index].item()) + "_" + str(counter) + "_" + str(
                        img_index) + ".png")

            elif self.mask_shape == "rotaEllipse":
                mask_save = np.zeros([MaskOut.shape[2], MaskOut.shape[3], 3])
                mask_save[:, :, 0] = MaskOut[img_index, 0, :, :]
                mask_save[:, :, 1] = MaskOut[img_index, 0, :, :]
                mask_save[:, :, 2] = MaskOut[img_index, 0, :, :]

                ellipse = np.ones([224, 224], dtype=np.uint8)
                ellipse = cv2.ellipse(ellipse, (int(OutLoc[img_index, 0] * 224), int(OutLoc[img_index, 1] * 224)),
                                 (int(OutLoc[img_index, 2] * 224 * 0.5), int(OutLoc[img_index, 3] * 224 * 0.5)),
                                 180 * OutLoc[img_index, 4], 0, 360, 0, 3)
                ellipse_index = np.where(ellipse < 0.5)
                ellipse_index = np.array(ellipse_index)
                ellipse_index = ellipse_index.transpose(1, 0)
                ellipse_index = ellipse_index[:, [1, 0]]
                box = cv2.boundingRect(ellipse_index)
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[0] + box[2])
                ymax = int(box[1] + box[3])

                per_img = np.zeros([224, 224, 3])
                per_img[:, :, 0] = saveImages[img_index, 0, :, :] * 255 * ellipse
                per_img[:, :, 1] = saveImages[img_index, 1, :, :] * 255 * ellipse
                per_img[:, :, 2] = saveImages[img_index, 2, :, :] * 255 * ellipse
                saveImg = Image.fromarray(per_img.astype(np.uint8))
                draw = ImageDraw.Draw(saveImg)

                #mask * image
                mask_img = np.zeros([MaskOut.shape[2], MaskOut.shape[3], 3])
                mask_img[:, :, 0] = saveImages[img_index, 0, :, :] * mask_save[:, :, 0] * 255
                mask_img[:, :, 1] = saveImages[img_index, 1, :, :] * mask_save[:, :, 1] * 255
                mask_img[:, :, 2] = saveImages[img_index, 2, :, :] * mask_save[:, :, 2] * 255
                mask_img = Image.fromarray(mask_img.astype(np.uint8))
                mask_img.save(
                    "temp/test/mask_" + str(labels[img_index].item()) + "_" + str(counter) + "_" + str(img_index) + self.loss_type+ ".png")

            draw.rectangle((int(xmin), int(ymin), int(xmax), int(ymax)), outline='blue', width=3)
            xmin = bboxes[img_index, 0].item()
            ymin = bboxes[img_index, 1].item()
            xmax = bboxes[img_index, 2].item()
            ymax = bboxes[img_index, 3].item()
            draw.rectangle((int(xmin), int(ymin), int(xmax), int(ymax)), outline='red', width=3)
            saveImg.save("temp/test/"+ str(labels[img_index].item()) + "_" + str(counter) + "_" + str(img_index) + self.loss_type+ ".png")
