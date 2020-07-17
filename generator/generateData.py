import torch
import torch.utils.data as data
import cv2
import numpy as np
import math

class generate_data(data.Dataset):
    def __init__(self, min_size, max_size, in_size, out_size, iteration, mode="rectangle"):
        self.min_size = min_size
        self.max_size = max_size
        self.in_size = in_size
        self.out_size = out_size
        self.iteration = iteration
        self.mode = mode
        # self.counter_0 = 0
        # self.counter_1 = 0
    def __getitem__(self, index):
        img, gt = self.pull_item(index)
        return img, gt

    def __len__(self):
        return self.iteration

    def pull_item(self, index):

        if self.mode == "rectangle":
            #generate bbox ratio
            rand_ratio = np.clip(np.random.normal(1, 0.5, 1), 0.2, 1.8)
            rand_size = np.clip(np.random.normal(0.5, 0.25, 1), self.min_size, self.max_size)
            # print("rand_ratio:", rand_ratio, "rand_size", rand_size)
            width = rand_size[0]
            height = rand_size[0] * rand_ratio[0]
            scale = min(self.max_size/max(height, width), 1)
            width = scale * width
            height = scale * height
            if width < self.min_size:
                width = self.min_size
            if height < self.min_size:
                height = self.min_size

            coord_min = np.random.rand(2) * np.array([1-width, 1-height]).reshape(2)
            coord_max = coord_min + np.array([width, height]).reshape(2)

            img = np.array([coord_min[0], coord_min[1], coord_max[0], coord_max[1]])
        elif self.mode == "rotaRectangle" or self.mode == "rotaEllipse":
            #generate bbox ratio
            rand_ratio = np.clip(np.random.normal(1, 0.9, 1), 0.2, 1.8)
            rand_size = np.clip(np.random.normal(0.5, 0.25, 1), self.min_size, self.max_size)
            # rota_angle = np.random.rand(1) - 0.5
            rota_angle = np.random.rand(1) - 0.5
            # print("rand_ratio:", rand_ratio, "rand_size", rand_size)
            if np.random.randint(2):
                width = rand_size[0]
                height = rand_size[0] * rand_ratio[0]
            else:
                width = rand_size[0] * rand_ratio[0]
                height = rand_size[0]
            scale = min(self.max_size/max(height, width), 1)
            width = scale * width
            height = scale * height
            if width < self.min_size:
                width = self.min_size
            if height < self.min_size:
                height = self.min_size

            coord_min = np.random.rand(2) * np.array([1-width, 1-height]).reshape(2)
            coord_max = coord_min + np.array([width, height]).reshape(2)

            coord_center = (coord_min + coord_max)/2

            while 1:
                bboxes = cv2.boxPoints(((coord_center[0]*self.out_size, coord_center[1]*self.out_size),
                                   (width*self.out_size, height*self.out_size), 180*rota_angle[0]))
                if np.max(bboxes) > self.out_size or np.min(bboxes) < 0:
                    width = width * 0.98
                    height = height * 0.98
                else:
                    break
            img = np.array([coord_center[0], coord_center[1], width, height, rota_angle[0]])
        elif self.mode == "quadrangle":
            #generate bbox ratio
            rand_ratio = np.clip(np.random.normal(1, 0.5, 1), 0.2, 1.8)
            rand_size = np.clip(np.random.normal(0.5, 0.25, 1), self.min_size, self.max_size)
            # print("rand_ratio:", rand_ratio, "rand_size", rand_size)
            width = rand_size[0]
            height = rand_size[0] * rand_ratio[0]
            scale = min(self.max_size/max(height, width), 1)
            width = scale * width
            height = scale * height
            if width < self.min_size:
                width = self.min_size
            if height < self.min_size:
                height = self.min_size

            coord_min = np.random.rand(2) * np.array([1-width, 1-height]).reshape(2)
            coord_max = coord_min + np.array([width, height]).reshape(2)


            # diagonal = width*width + height*height

            #generate first point
            P1_x = coord_min[0] + np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*width
            P1_y = coord_min[1] + np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*height
            P2_x = coord_max[0] - np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*width
            P2_y = coord_min[1] + np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*height
            P3_x = coord_max[0] - np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*width
            P3_y = coord_max[1] - np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*height
            P4_x = coord_min[0] + np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*width
            P4_y = coord_max[1] - np.clip(np.random.normal(0, 0.2, 1), 0, 0.5)*height
            # print(P1_x[0], P1_y[0], P2_x[0], P2_y[0], P3_x[0], P3_y[0], P4_x[0], P4_y[0])
            img = np.array([P1_x[0], P1_y[0], P2_x[0], P2_y[0], P3_x[0], P3_y[0], P4_x[0], P4_y[0]])


        #generate gt
        gt = np.zeros([self.out_size, self.out_size], dtype=np.float)
        if self.mode == "rectangle":
            xmin = int(coord_min[0] * self.out_size)
            ymin = int(coord_min[1] * self.out_size)
            xmax = int(coord_max[0] * self.out_size)
            ymax = int(coord_max[1] * self.out_size)
            # print(xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin)
            bboxes =  np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
            gt = cv2.drawContours(gt, [bboxes], -1, 1, -1)
        elif self.mode == "rotaRectangle":
            # xmin = int(coord_min[0] * self.out_size)
            # ymin = int(coord_min[1] * self.out_size)
            # xmax = int(coord_max[0] * self.out_size)
            # ymax = int(coord_max[1] * self.out_size)
            # # print(xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin)
            # bboxes =  np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
            # gt = cv2.drawContours(gt, [bboxes], -1, 125, -1)

            center = (coord_min + coord_max)/2
            bboxes = cv2.boxPoints(((center[0]*self.out_size, center[1]*self.out_size),
                               (width*self.out_size, height*self.out_size), 180*rota_angle[0]))
            bboxes = np.int0(bboxes)
            gt = cv2.drawContours(gt, [bboxes], -1, 1, -1)
        elif self.mode == "rotaEllipse":
            center = (coord_min + coord_max)/2
            # bboxes = cv2.boxPoints(((center[0]*self.out_size, center[1]*self.out_size),
            #                    (width*self.out_size, height*self.out_size), 90*rota_angle[0]))
            # bboxes = np.int0(bboxes)
            # gt = cv2.drawContours(gt, [bboxes], -1, 125, -1)
            gt = cv2.ellipse(gt, (int(center[0]*self.out_size), int(center[1]*self.out_size)),
                             (int(width*self.out_size/2), int(height*self.out_size/2)), 180*rota_angle[0], 0, 360, 1, -1)


        elif self.mode == "quadrangle":
            gt = np.zeros([self.out_size, self.out_size], dtype=np.float)
            P1_x = int(P1_x[0] * self.out_size)
            P1_y = int(P1_y[0] * self.out_size)
            P2_x = int(P2_x[0] * self.out_size)
            P2_y = int(P2_y[0] * self.out_size)
            P3_x = int(P3_x[0] * self.out_size)
            P3_y = int(P3_y[0] * self.out_size)
            P4_x = int(P4_x[0] * self.out_size)
            P4_y = int(P4_y[0] * self.out_size)
            # print(xmin, ymin, xmax, ymax, xmax - xmin, ymax - ymin)
            bboxes =  np.array([[P1_x, P1_y], [P2_x, P2_y], [P3_x, P3_y], [P4_x, P4_y]])
            gt = cv2.drawContours(gt, [bboxes], -1, 1, -1)


        # #visualization
        # visiImg = np.zeros([self.out_size, self.out_size, 3], dtype=np.float)
        # visiImg[:, :, 0] = gt
        # visiImg[:, :, 1] = gt
        # visiImg[:, :, 2] = gt
        # # if self.mode == "rectangle":
        # #     left_up, right_bottom = (xmin, ymin), (xmax, ymax)
        # #     cv2.rectangle(visiImg, left_up, right_bottom, (0, 255, 0), 2)
        # # elif self.mode == "quadrangle":
        # #     cv2.circle(visiImg, (P1_x, P1_y), 1, (0, 0, 255), 4)
        # #     cv2.circle(visiImg, (P2_x, P2_y), 1, (0, 255, 0), 4)
        # #     cv2.circle(visiImg, (P3_x, P3_y), 1, (255, 0, 0), 4)
        # #     cv2.circle(visiImg, (P4_x, P4_y), 1, (0, 255, 255), 4)
        # cv2.imwrite("temp/"+str(index)+".jpg", visiImg)

        return torch.from_numpy(img).float(), torch.from_numpy(gt.reshape([1, self.out_size, self.out_size])).float()

# # "rectangle, rotaRectangle, rotaEllipse, quadrangle"
# test = generate_data(0.05, 0.8, 12, 224, 100, "quadrangle")
# for index in range(1000):
#     test.__getitem__(index)
