import numpy as np
import cv2
import scipy.io, scipy.signal
import argparse
from modules.network import UNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from PIL import Image
from skimage import img_as_bool

class irisRecognition(object):
    def __init__(self, cfg):

        # cParsing the config file
        self.height = cfg["polar_height"]
        self.width = cfg["polar_width"]
        self.angles = angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.width)
        self.cos_angles = np.zeros((self.width))
        self.sin_angles = np.zeros((self.width))
        for i in range(self.width):
            self.cos_angles[i] = np.cos(self.angles[i])
            self.sin_angles[i] = np.sin(self.angles[i])
        self.filter_size = cfg["recog_filter_size"]
        self.num_filters = cfg["recog_num_filters"]
        self.max_shift = cfg["recog_max_shift"]
        self.cuda = cfg["cuda"]
        self.cuda = cfg["gpu"]
        self.ccnet_model_path = cfg["ccnet_model_path"]
        self.filter = scipy.io.loadmat(cfg["recog_bsif_dir"]+'ICAtextureFilters_{0}x{1}_{2}bit.mat'.format(self.filter_size, self.filter_size, self.num_filters))['ICAtextureFilters']
        self.iris_hough_param1 = cfg["iris_hough_param1"]
        self.iris_hough_param2 = cfg["iris_hough_param2"]
        self.iris_hough_margin = cfg["iris_hough_margin"]
        self.pupil_hough_param1 = cfg["pupil_hough_param1"]
        self.pupil_hough_param2 = cfg["pupil_hough_param2"]
        self.pupil_hough_minimum = cfg["pupil_hough_minimum"]
        self.pupil_iris_max_ratio = cfg["pupil_iris_max_ratio"]
        self.max_pupil_iris_shift = cfg["max_pupil_iris_shift"]

        # Loading the CCNet
        self.CCNET_INPUT_SIZE = (320,240)
        self.CCNET_NUM_CHANNELS = 1
        self.CCNET_NUM_CLASSES = 2
        self.model = UNet(self.CCNET_NUM_CLASSES, self.CCNET_NUM_CHANNELS)
        if self.cuda:
            torch.cuda.set_device(self.gpu)
            self.model = model.cuda()
        if self.ccnet_model_path:
            try:
                if self.cuda:
                    self.model.load_state_dict(torch.load(self.ccnet_model_path))
                else:
                    self.model.load_state_dict(torch.load(self.ccnet_model_path, map_location=torch.device('cpu')))
                    # print("model state loaded")
            except AssertionError:
                print("assertion error")
                self.model.load_state_dict(torch.load(self.ccnet_model_path,
                    map_location = lambda storage, loc: storage))
        self.model.eval()
        self.softmax = nn.LogSoftmax(dim=1)
        self.input_transform = Compose([ToTensor(),])

        print("irisRecognition class: initialized")


    def segment(self,image):

        w,h = image.size
        image = cv2.resize(np.array(image), self.CCNET_INPUT_SIZE, cv2.INTER_CUBIC)

        outputs = self.model(Variable(self.input_transform(image).unsqueeze(0)))
        logprob = self.softmax(outputs).data.cpu().numpy()
        pred = np.argmax(logprob, axis=1)*255
        pred = Image.fromarray(pred[0].astype(np.uint8))
        pred = np.array(pred)

        # Optional: uncomment the following lines to take only the biggest blob returned by CCNet
        '''
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=4)
        if nb_components > 1:
            sizes = stats[:, -1] 
            max_label = 1
            max_size = sizes[1]    
            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]

            pred = np.zeros(output.shape)
            pred[output == max_label] = 255
            pred = np.asarray(pred, dtype=np.uint8)
        '''

        # Resize the mask to the original image size
        pred = img_as_bool(cv2.resize(np.array(pred), (w,h), cv2.INTER_NEAREST))

        return pred


    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):

        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + 96*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis

    def circApprox(self,mask):

        # Iris boundary approximation
        mask_for_iris = 255*(1 - np.uint8(mask))
        iris_indices = np.where(mask_for_iris == 0)
        if len(iris_indices[0]) == 0:
            return None, None
        y_span = max(iris_indices[0]) - min(iris_indices[0])
        x_span = max(iris_indices[1]) - min(iris_indices[1])

        iris_radius_estimate = np.max((x_span,y_span)) // 2
        iris_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=self.iris_hough_param1,
                                       param2=self.iris_hough_param2,
                                       minRadius=iris_radius_estimate-self.iris_hough_margin,
                                       maxRadius=iris_radius_estimate+self.iris_hough_margin)
        if iris_circle is None:
            return None, None
        iris_x, iris_y, iris_r = np.rint(np.array(iris_circle[0][0])).astype(int)
        
        
        # Pupil boundary approximation
        pupil_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                        param1=self.pupil_hough_param1,
                                        param2=self.pupil_hough_param2,
                                        minRadius=self.pupil_hough_minimum,
                                        maxRadius=np.int(self.pupil_iris_max_ratio*iris_r))
        if pupil_circle is None:
            return None, None
        pupil_x, pupil_y, pupil_r = np.rint(np.array(pupil_circle[0][0])).astype(int)
        
        if np.sqrt((pupil_x-iris_x)**2+(pupil_y-iris_y)**2) > self.max_pupil_iris_shift:
            pupil_x = iris_x
            pupil_y = iris_y
            pupil_r = iris_r // 3
        
        return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])



    # Rubbersheet model-based Cartesian-to-polar transformation
    def cartToPol(self, image, cx, cy, pupil_r, iris_r):
       
        polar = np.zeros((self.height, self.width), np.uint8)

        for j in range(self.height):
            rad = j /self.height

            x_lowers = cx + pupil_r * self.cos_angles
            y_lowers = cy + pupil_r * self.sin_angles
            x_uppers = cx + iris_r * self.cos_angles
            y_uppers = cy + iris_r * self.sin_angles

            Xc = (1 - rad) * x_lowers + rad * x_uppers
            Yc = (1 - rad) * y_lowers + rad * y_uppers

            polar[j, :] = image[Xc.astype(int), Yc.astype(int)]

        return polar


    # Iris code
    def extractCode(self, polar):
        
        # Wrap image
        r = int(np.floor(self.filter_size / 2));
        imgWrap = np.zeros((r*2+self.height, r*2+self.width))
        imgWrap[:r, :r] = polar[-r:, -r:]
        imgWrap[:r, r:-r] = polar[-r:, :]
        imgWrap[:r, -r:] = polar[-r:, :r]

        imgWrap[r:-r, :r] = polar[:, -r:]
        imgWrap[r:-r, r:-r] = polar
        imgWrap[r:-r, -r:] = polar[:, :r]

        imgWrap[-r:, :r] = polar[:r, -r:]
        imgWrap[-r:, r:-r] = polar[:r, :]
        imgWrap[-r:, -r:] = polar[:r, :r]

        # Loop over all BSIF kernels in the filter set
        codeBinary = np.zeros((self.height, self.width, self.num_filters))
        for i in range(1,self.num_filters+1):
            ci = scipy.signal.convolve2d(imgWrap, np.rot90(self.filter[:,:,self.num_filters-i],2), mode='valid')
            codeBinary[:,:,i-1] = ci>0

        return codeBinary

    # Match iris codes
    def matchCodes(self, code1, code2, mask1, mask2):
        margin = int(np.ceil(self.filter_size/2))
        self.code1 = code1[margin:-margin, :, :]
        self.code2 = code2[margin:-margin, :, :]
        self.mask1 = mask1[margin:-margin, :]
        self.mask2 = mask2[margin:-margin, :]

        scoreC = np.zeros((self.num_filters, 2*self.max_shift+1))
        for shift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(self.mask1, np.roll(self.mask2, shift, axis=1))
            xorCodes = np.logical_xor(self.code1, np.roll(self.code2, shift, axis=1))
            xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=2),self.num_filters))
            scoreC[:,shift] = np.sum(xorCodesMasked, axis=(0,1)) / np.sum(andMasks)

        scoreC = np.min(np.mean(scoreC, axis=0))

        return scoreC
