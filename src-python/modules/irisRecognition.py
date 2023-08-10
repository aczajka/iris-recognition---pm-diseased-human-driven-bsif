import numpy as np
import cv2
import scipy.io, scipy.signal
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from math import pi
from torchvision import models
from modules.network import UNet, NestedSharedAtrousResUNet

class irisRecognition(object):
    def __init__(self, cfg, use_hough=True):
        

        # cParsing the config file
        self.use_hough = use_hough
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.filter_size = cfg["recog_filter_size"]
        self.num_filters = cfg["recog_num_filters"]
        self.max_shift = cfg["recog_max_shift"]
        self.cuda = cfg["cuda"]
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.mask_model_path = cfg["mask_model_path"]
        self.circle_model_path = cfg["circle_model_path"]
        self.ccnet_model_path = cfg["ccnet_model_path"]
        #filter_mat = scipy.io.loadmat(cfg["recog_bsif_dir"]+'ICAtextureFilters_{0}x{1}_{2}bit.mat'.format(self.filter_size, self.filter_size, self.num_filters))['ICAtextureFilters']
        mat_file_path = cfg["recog_bsif_dir"]+'ICAtextureFilters_{0}x{1}_{2}bit.pt'.format(self.filter_size, self.filter_size, self.num_filters)
        with torch.no_grad():
            filter_mat = torch.jit.load(mat_file_path, torch.device('cpu')).ICAtextureFilters.detach().numpy()
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        with torch.no_grad():
            self.torch_filter = torch.FloatTensor(filter_mat).to(self.device)
            self.torch_filter = torch.moveaxis(self.torch_filter.unsqueeze(0), 3, 0).detach().requires_grad_(False)
        if self.use_hough == True:
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
        self.NET_INPUT_SIZE = (320,240)
        # Loading the CCNet
        if not self.use_hough:
            if self.mask_model_path and self.circle_model_path:
                self.circle_model = models.resnet50()
                self.circle_model.fc = nn.Sequential(
                        nn.Linear(2048, 64),
                        nn.ReLU(inplace=True),
                        nn.Linear(64, 6)
                )
                try:
                    self.circle_model.load_state_dict(torch.load(self.circle_model_path, map_location=self.device))
                except AssertionError:
                        print("assertion error")
                        self.circle_model.load_state_dict(torch.load(self.circle_model_path,
                            map_location = lambda storage, loc: storage))
                self.circle_model = self.circle_model.to(self.device)
                self.circle_model.eval()
                self.mask_model = NestedSharedAtrousResUNet(1, 1, width=64, resolution=(240,320))
                try:
                    self.mask_model.load_state_dict(torch.load(self.mask_model_path, map_location=self.device))
                except AssertionError:
                        print("assertion error")
                        self.mask_model.load_state_dict(torch.load(self.mask_model_path,
                            map_location = lambda storage, loc: storage))
                self.mask_model = self.mask_model.to(self.device)
                self.mask_model.eval()
                self.input_transform_mask = Compose([
                    ToTensor(),
                    #Normalize(mean=(0.5791223733793273,), std=(0.21176097694558188,))
                    Normalize(mean=(0.5,), std=(0.5,))
                ])
                self.input_transform_circ = Compose([
                    ToTensor(),
                    #Normalize(mean=(0.5791223733793273,), std=(0.21176097694558188,))
                    Normalize(mean=(0.5,), std=(0.5,))
                ])
        else:
            self.CCNET_NUM_CHANNELS = 1
            self.CCNET_NUM_CLASSES = 2
            self.model = UNet(self.CCNET_NUM_CLASSES, self.CCNET_NUM_CHANNELS)
            if self.ccnet_model_path:
                try:
                    if self.cuda:
                        self.model.load_state_dict(torch.load(self.ccnet_model_path, map_location=torch.device('cuda')))
                    else:
                        self.model.load_state_dict(torch.load(self.ccnet_model_path, map_location=torch.device('cpu')))
                        print("model state loaded")
                except AssertionError:
                    print("assertion error")
                    self.model.load_state_dict(torch.load(self.ccnet_model_path,
                        map_location = lambda storage, loc: storage))
            else:
                print("Please provide the CCNet model weight.")
            self.model.eval()
            
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.input_transform = Compose([ToTensor(),])

        # Misc
        self.se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        self.ISO_RES = (640,480)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))

    # converts non-ISO images into ISO dimensions
    def fix_image(self, image):
        w, h = image.size
        aspect_ratio = float(w)/float(h)
        if aspect_ratio >= 1.333 and aspect_ratio <= 1.334:
            result_im = image.resize((640, 480))
        elif aspect_ratio < 1.333:
            w_new = h * (4.0/3.0)
            w_pad = (w_new - w) / 2
            result_im = Image.new(image.mode, (int(w_new), h), 127)
            result_im.paste(image, (int(w_pad), 0))
            result_im = result_im.resize((640, 480))
        else:
            h_new = w * (3.0/4.0)
            h_pad = (h_new - h) / 2
            result_im = Image.new(image.mode, (w, int(h_new)), 127)
            result_im.paste(image, (0, int(h_pad)))
            result_im = result_im.resize((640, 480))
        return result_im
    
    ### Use this function for a faster estimation of mask and circle parameters. When use_hough is False, this function uses both the circle approximation and the mask from MCCNet
    def segment_and_circApprox(self, image):
        pred = self.segment(image)
        pupil_xyr, iris_xyr = self.circApprox(pred, image)
        return pred, pupil_xyr, iris_xyr

    def segment(self,image):
        if self.use_hough:
            w,h = image.size
            image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR)
            with torch.no_grad():
                outputs = self.model(Variable(self.input_transform(image).unsqueeze(0).to(self.device)))
            logprob = self.softmax(outputs).data.cpu().numpy()
            pred = np.argmax(logprob, axis=1)*255
            pred = Image.fromarray(pred[0].astype(np.uint8), 'L')
            # Resize the mask to the original image size
            pred = cv2.resize(np.uint8(pred), (w,h), cv2.INTER_NEAREST_EXACT)

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
            return pred
        else:
            w,h = image.size
            image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_CUBIC)
            with torch.no_grad():
                mask_logit_t = self.mask_model(Variable(self.input_transform_mask(image).unsqueeze(0).to(self.device)))[0]
                mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
            mask = mask_t.cpu().numpy()[0]
            mask = cv2.resize(np.uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
            #print('Mask Shape: ', mask.shape)

            return mask


    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):

        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + (96/255)*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis


    def circApprox(self,mask=None,image=None):
        if self.use_hough and mask is None:
            print('Please provide mask if you want to use hough transform')
        if (not self.use_hough) and image is None:
            print('Please provide image if you want to use the mccnet model') 
        if self.use_hough and (mask is not None):
            # Iris boundary approximation
            mask_for_iris = cv2.bitwise_not(mask)
            iris_indices = np.nonzero(mask)
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
        elif (not self.use_hough) and (image is not None):
            w,h = image.size

            image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_CUBIC)
            with torch.no_grad():
                inp_xyr_t = self.circle_model(Variable(self.input_transform_circ(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))

            #Circle params
            diag = math.sqrt(w**2 + h**2)
            inp_xyr = inp_xyr_t.tolist()[0]
            pupil_x = int(inp_xyr[0] * w)
            pupil_y = int(inp_xyr[1] * h)
            pupil_r = int(inp_xyr[2] * 0.5 * 0.8 * diag)
            iris_x = int(inp_xyr[3] * w)
            iris_y = int(inp_xyr[4] * h)
            iris_r = int(inp_xyr[5] * 0.5 * diag)

            return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
    
    # New Rubbersheet model-based Cartesian-to-polar transformation. It has support for bilinear interpolation
    def grid_sample(self, input, grid, interp_mode):  #helper function for new interpolation

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=True)

    def cartToPol_torch(self, image, mask, pupil_xyr, iris_xyr, interpolation='bilinear'): # 
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = torch.tensor(np.array(image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
            mask = torch.tensor(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(self.device)
            width = image.shape[3]
            height = image.shape[2]

            polar_height = self.polar_height
            polar_width = self.polar_width
            pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float().to(self.device)
            iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float().to(self.device)
            
            theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).to(self.device)
            pxCirclePoints = torch.round(pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device) #b x 512
            pyCirclePoints = torch.round(pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device)  #b x 512
            
            ixCirclePoints = torch.round(iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device)  #b x 512
            iyCirclePoints = torch.round(iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device) #b x 512

            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).to(self.device)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

            x = (pxCoords + ixCoords).float()
            x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512

            y = (pyCoords + iyCoords).float()
            y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512

            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).to(self.device)

            image_polar = self.grid_sample(image, grid_sample_mat, interp_mode=interpolation)
            image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest') #no use using bilinear for interpolation in mask
            mask_polar = (mask_polar>0.5).long() * 255

            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)
    
    # (Fixed) Old Rubbersheet model-based Cartesian-to-polar transformation uses nearest neighbor interpolation
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(0,self.polar_width-1,self.polar_width)/self.polar_width
       
        pxCirclePoints = np.around(pupil_xyr[0] + pupil_xyr[2]*np.cos(theta))    
        ixCirclePoints = np.around(iris_xyr[0] + iris_xyr[2]*np.cos(theta))
        pyCirclePoints = np.around(pupil_xyr[1] + pupil_xyr[2]*np.sin(theta))
        iyCirclePoints = np.around(iris_xyr[1] + iris_xyr[2]*np.sin(theta))

        for j in range(1, self.polar_width+1):            
            for i in range(1, self.polar_height+1):

                radius = i/self.polar_height
                x = int(np.around((1-radius) * pxCirclePoints[j-1] + radius * ixCirclePoints[j-1]))
                y = int(np.around((1-radius) * pyCirclePoints[j-1] + radius * iyCirclePoints[j-1]))
                if (x > 0 and x <= width and y > 0 and y <= height): 
                    image_polar[i-1][j-1] = image[y-1][x-1]
                    mask_polar[i-1][j-1] = mask[y-1][x-1]

        return image_polar, mask_polar
    
    '''
    # Previous implementation which is removed due to mismatch with MATLAB
    # Rubbersheet model-based Cartesian-to-polar transformation
    def cartToPol_prev(self, image, mask, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
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
    
    # Previous implementation of iris code extraction, updated code only wraps horizontally
    # Iris code
    def extractCode(self, polar):
        
        if polar is None:
            return None
        
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
    '''

    def extractCode(self, polar):
        with torch.no_grad():
            if polar is None:
                return None
            r = int(np.floor(self.filter_size / 2))
            polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0)
            polar_t = (polar_t - polar_t.min()) / (polar_t.max() - polar_t.min())
            padded_polar = nn.functional.pad(polar_t, (r, r, 0, 0), mode='circular')
            codeContinuous = nn.functional.conv2d(padded_polar, self.torch_filter)
            codeBinary = torch.where(codeContinuous.squeeze(0) > 0, True, False).cpu().numpy()
            return codeBinary # The size of the code should be: 7 x 48 x 512 
    
    def matchCodes(self, code1, code2, mask1, mask2):
        r = int(np.floor(self.filter_size / 2))
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1[r:-r, :] > 127, True, False) 
        mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
        #print(mask1_binary.shape, mask2_binary.shape)
        scoreC = np.zeros((self.num_filters, 2*self.max_shift+1))
        for shift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, shift, axis=1))
            xorCodes = np.logical_xor(code1, np.roll(code2, shift, axis=2))
            xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.num_filters, 1, 1)))
            scoreC[:,shift] = np.sum(xorCodesMasked, axis=(1,2)) / np.sum(andMasks)
        
        scoreMean = np.mean(scoreC, axis=0)
        scoreC = np.min(scoreMean)
        scoreC_shift = self.max_shift - np.argmin(scoreMean)
        return scoreC, scoreC_shift

    def polar(self,x,y):
        return math.hypot(x,y),math.degrees(math.atan2(y,x))

    def visualizeMatchingResult(self, code1, code2, mask1, mask2, shift, im, pupil_xyr, iris_xyr):
        
        resMask = np.zeros((self.ISO_RES[1],self.ISO_RES[0]))

        r = int(np.floor(self.filter_size / 2))
        # calculate heat map
        xorCodes = np.logical_xor(code1, np.roll(code2, self.max_shift-shift, axis=1))
        mask1_binary = np.where(mask1[r:-r, :] > 127, True, False) 
        mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
        andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, shift, axis=1))

        heatMap = 1-xorCodes.astype(int)
        heatMap = np.pad(np.mean(heatMap,axis=0), pad_width=((8,8),(0,0)), mode='constant', constant_values=0)
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
