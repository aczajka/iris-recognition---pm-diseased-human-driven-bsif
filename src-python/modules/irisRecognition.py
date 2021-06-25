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
import math
from math import pi

class irisRecognition(object):
    def __init__(self, cfg):

        # cParsing the config file
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.angles = angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.polar_width)
        self.cos_angles = np.zeros((self.polar_width))
        self.sin_angles = np.zeros((self.polar_width))
        for i in range(self.polar_width):
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
        self.visMinAgreedBits = cfg["vis_min_agreed_bits"]
        self.vis_mode = cfg["vis_mode"]

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

        # Misc
        self.se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        self.ISO_RES = (640,480)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))

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
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(1,self.polar_width,self.polar_width)/self.polar_width
        pxCirclePoints = pupil_xyr[0] + pupil_xyr[2]*np.cos(theta)
        pyCirclePoints = pupil_xyr[1] + pupil_xyr[2]*np.sin(theta)
        
        ixCirclePoints = iris_xyr[0] + iris_xyr[2]*np.cos(theta)
        iyCirclePoints = iris_xyr[1] + iris_xyr[2]*np.sin(theta)

        radius = np.linspace(0,self.polar_height,self.polar_height)/self.polar_height
        for j in range(self.polar_width):
            x = (np.clip(0,width-1,np.around((1-radius) * pxCirclePoints[j] + radius * ixCirclePoints[j]))).astype(int)
            y = (np.clip(0,height-1,np.around((1-radius) * pyCirclePoints[j] + radius * iyCirclePoints[j]))).astype(int)
            
            for i in range(self.polar_height):
                if (x[i] > 0 and x[i] < width and y[i] > 0 and y[i] < height): 
                    image_polar[i][j] = image[y[i]][x[i]]
                    mask_polar[i][j] = 255*mask[y[i]][x[i]]

        return image_polar, mask_polar


    # Iris code
    def extractCode(self, polar):
        
        # Wrap image
        r = int(np.floor(self.filter_size / 2));
        imgWrap = np.zeros((r*2+self.polar_height, r*2+self.polar_width))
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
        codeBinary = np.zeros((self.polar_height, self.polar_width, self.num_filters))
        for i in range(1,self.num_filters+1):
            ci = scipy.signal.convolve2d(imgWrap, np.rot90(self.filter[:,:,self.num_filters-i],2), mode='valid')
            codeBinary[:,:,i-1] = ci>0

        return codeBinary

    # Match iris codes
    def matchCodes(self, code1, code2, mask1, mask2):
        
        if code1 is None or mask1 is None:
            return -1., 0.
        if code2 is None or mask2 is None:
            return -2., 0.

        margin = int(np.ceil(self.filter_size/2))
        code1 = np.array(code1)
        code2 = np.array(code2)
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        
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

        scoreMean = np.mean(scoreC, axis=0)
        scoreC = np.min(scoreMean)
        scoreC_shift = self.max_shift-np.argmin(scoreMean)

        return scoreC, scoreC_shift

    def polar(self,x,y):
        return math.hypot(x,y),math.degrees(math.atan2(y,x))



    def visualizeMatchingResult(self, code1, code2, mask1, mask2, shift, im, pupil_xyr, iris_xyr):
        
        resMask = np.zeros((self.ISO_RES[1],self.ISO_RES[0]))

        # calculate heat map
        xorCodes = np.logical_xor(self.code1, np.roll(self.code2, self.max_shift-shift, axis=1))
        andMasks = np.logical_and(self.mask1, np.roll(self.mask2, self.max_shift-shift, axis=1))

        heatMap = 1-xorCodes.astype(int)
        heatMap = np.pad(np.mean(heatMap,axis=2), pad_width=((8,8),(0,0)), mode='constant', constant_values=0)
        andMasks = np.pad(andMasks, pad_width=((8,8),(0,0)), mode='constant', constant_values=0)
        heatMap = heatMap * andMasks

        if 'single' in self.vis_mode:
            heatMap = (heatMap >= self.visMinAgreedBits / 100).astype(np.uint8)

        heatMap = np.roll(heatMap,int(self.polar_width/2),axis=1)

        for j in range(self.ISO_RES[0]):
            for i in range(self.ISO_RES[1]):
                xi = j-iris_xyr[0]
                yi = i-iris_xyr[1]
                ri = iris_xyr[2]
                xp = j-pupil_xyr[0]
                yp = i-pupil_xyr[1]
                rp = pupil_xyr[2]

                if xi**2 + yi**2 < ri**2 and xp**2 + yp**2 > rp**2:
                    rr,tt = self.polar(xi,yi)
                    tt = np.clip(np.round(self.polar_width*((180+tt)/360)).astype(int),0,self.polar_width-1)
                    rr = np.clip(np.round(self.polar_height * (rr - rp) / (ri - rp)).astype(int),0,self.polar_height-1)
                    resMask[i,j] = heatMap[rr,tt] # *** TODO correct mapping for shifted p/i centers 
        
        heatMap = 255*cv2.morphologyEx(resMask, cv2.MORPH_OPEN, kernel=self.se)
        mask_blur = cv2.filter2D(heatMap,-1,self.sk)

        if 'single' in self.vis_mode:
            mask_blur = (48 * mask_blur / np.max(mask_blur)).astype(int)
            imVis = np.stack((np.array(im),)*3, axis=-1)
            imVis[:,:,1] = np.clip(imVis[:,:,1] + mask_blur,0,255)
        elif 'heat_map' in self.vis_mode:
            mask_blur = (255 * mask_blur / np.max(mask_blur)).astype(int)
            heatMap = np.uint8(np.stack((np.array(mask_blur),)*3, axis=-1))
            heatMap = cv2.applyColorMap(heatMap, cv2.COLORMAP_JET)
            cl_im = self.clahe.apply(np.array(im))
            imVis = np.stack((cl_im,)*3, axis=-1)
            imVis = cv2.addWeighted(heatMap, 0.1, np.array(imVis), 0.9, 32)
        else:
            raise Exception("Unknown visualization mode")

        return imVis

