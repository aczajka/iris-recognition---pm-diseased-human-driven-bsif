import numpy as np
import cv2
import scipy.io, scipy.signal
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math
from math import pi
from torchvision import models
from modules.network import *
import multiprocessing

def matchCodesGlobal(irisRecObj, code1, code2, mask1, mask2): #global function required for multiprocessing
    r = int(np.floor(irisRecObj.filter_size / 2))
    # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
    mask1_binary = np.where(mask1[r:-r, :] > 127, True, False) 
    mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
    if (np.sum(mask1_binary) <= irisRecObj.threshold_frac_avg_bits * irisRecObj.avg_num_bits) or (np.sum(mask2_binary) <= irisRecObj.threshold_frac_avg_bits * irisRecObj.avg_num_bits):
        return -1.0
    scoreC = []
    for xshift in range(-irisRecObj.max_shift, irisRecObj.max_shift+1):
        andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
        if np.sum(andMasks) == 0:
            scoreC.append(float('inf'))
        else:
            xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
            xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (irisRecObj.num_filters, 1, 1)))
            scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * irisRecObj.num_filters))
        if irisRecObj.score_norm:
            scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( np.sum(andMasks) / irisRecObj.avg_num_bits )
    scoreC_best = np.min(np.array(scoreC))
    if scoreC_best == float('inf'):
        return -1.0
    return scoreC_best

class irisRecognition(object):
    def __init__(self, cfg, use_hough = False):
        # cParsing the config file
        self.jitter = 1
        self.use_hough = use_hough
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.filter_size = cfg["recog_filter_size"]
        self.num_filters = cfg["recog_num_filters"]
        self.max_shift = cfg["recog_max_shift"]
        self.cuda = cfg["cuda"]
        self.score_norm = cfg["score_normalization"]
        self.threshold_frac_avg_bits = cfg["threshold_frac_avg_bits"]
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
        #filter_mat_scipy = scipy.io.loadmat('../filters/finetuned_bsif_eyetracker_data/'+'ICAtextureFilters_{0}x{1}_{2}bit.mat'.format(self.filter_size, self.filter_size, self.num_filters))['ICAtextureFilters']
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        with torch.no_grad():
            self.torch_filter = torch.FloatTensor(filter_mat).to(self.device)
            self.torch_filter = torch.moveaxis(self.torch_filter.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            #self.torch_filter_scipy = torch.FloatTensor(filter_mat_scipy).to(self.device)
            #self.torch_filter_scipy = torch.moveaxis(self.torch_filter_scipy.unsqueeze(0), 3, 0).detach().requires_grad_(False)
        if self.use_hough:
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
                self.circle_model = models.convnext_tiny()
                self.circle_model.avgpool = conv(in_channels=768, out_n=6)
                self.circle_model.classifier = fclayer(in_h=7, in_w=10, out_n=6)
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
                    Normalize(mean=(0.5,), std=(0.5,))
                ])
                self.input_transform_circ = Compose([
                    ToTensor(),
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

        avg_bits_by_filter_size = {5: 25056, 7: 24463, 9: 23764, 11: 23010, 13: 22225, 15: 21420, 17: 20603, 19: 19777, 21: 18945, 27: 16419, 33: 13864, 39: 11289}
        self.avg_num_bits = avg_bits_by_filter_size[self.filter_size]
        self.ISO_RES = (640,480)
        self.multiproc = cfg["use_multiprocessing_jitter"]
        self.num_workers = cfg["num_workers"]

        self.se = np.ones((3, 3), dtype="uint8")
        self.sk = np.ones((3, 3), dtype="uint8")
        self.clahe = cv2.createCLAHE(clipLimit=5)

    # converts non-ISO images into ISO dimensions
    def fix_image(self, image):
        w, h = image.size
        aspect_ratio = float(w)/float(h)
        if aspect_ratio >= 1.333 and aspect_ratio <= 1.334:
            result_im = image.copy().resize(self.ISO_RES)
        elif aspect_ratio < 1.333:
            w_new = h * (4.0/3.0)
            w_pad = (w_new - w) / 2
            result_im = Image.new(image.mode, (int(w_new), h), 127)
            result_im.paste(image, (int(w_pad), 0))
            result_im = result_im.resize(self.ISO_RES)
        else:
            h_new = w * (3.0/4.0)
            h_pad = (h_new - h) / 2
            result_im = Image.new(image.mode, (w, int(h_new)), 127)
            result_im.paste(image, (0, int(h_pad)))
            result_im = result_im.resize(self.ISO_RES)
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
            image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
            with torch.no_grad():
                mask_logit_t = self.mask_model(Variable(self.input_transform_mask(image).unsqueeze(0).to(self.device)))[0]
                mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
            mask = mask_t.cpu().numpy()[0]
            mask = cv2.resize(np.uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
            #print('Mask Shape: ', mask.shape)

            return mask


    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):
        pupil_xyr = np.around(pupil_xyr).astype(np.int32)
        iris_xyr = np.around(iris_xyr).astype(np.int32)
        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + (96/255)*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis


    def circApprox(self,mask=None,image=None):
        if self.use_hough and (mask is None):
            print('Please provide mask if you want to use hough transform')
        if (not self.use_hough) and (image is None):
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
            iris_x, iris_y, iris_r = np.round(np.array(iris_circle[0][0])).astype(int)
            
            
            # Pupil boundary approximation
            pupil_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                            param1=self.pupil_hough_param1,
                                            param2=self.pupil_hough_param2,
                                            minRadius=self.pupil_hough_minimum,
                                            maxRadius=np.int32(self.pupil_iris_max_ratio*iris_r))
            if pupil_circle is None:
                return None, None
            pupil_x, pupil_y, pupil_r = np.round(np.array(pupil_circle[0][0])).astype(int)
            
            if np.sqrt((pupil_x-iris_x)**2+(pupil_y-iris_y)**2) > self.max_pupil_iris_shift:
                pupil_x = iris_x
                pupil_y = iris_y
                pupil_r = iris_r // 3

            return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
        elif (not self.use_hough) and (image is not None):
            w,h = image.size

            image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
            with torch.no_grad():
                inp_xyr_t = self.circle_model(Variable(self.input_transform_circ(image).unsqueeze(0).repeat(1,3,1,1).to(self.device)))

            #Circle params
            diag = math.sqrt(w**2 + h**2)
            inp_xyr = inp_xyr_t.tolist()[0]
            pupil_x = (inp_xyr[0] * w)
            pupil_y = (inp_xyr[1] * h)
            pupil_r = (inp_xyr[2] * 0.5 * 0.8 * diag)
            iris_x = (inp_xyr[3] * w)
            iris_y = (inp_xyr[4] * h)
            iris_r = (inp_xyr[5] * 0.5 * diag)

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
            pxCirclePoints = (pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device) #b x 512
            pyCirclePoints = (pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device)  #b x 512
            
            ixCirclePoints = (iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device)  #b x 512
            iyCirclePoints = (iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device) #b x 512

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
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest') # always use nearest neighbor interpolation for mask
            mask_polar = (mask_polar>0.5).long() * 255

            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)
    
    def cartToPol_torch_jitter(self, image, mask, pupil_xyr, iris_xyr, interpolation='bilinear'): # 
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
            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).to(self.device)  #64 x 1
            
            images_polar = []
            masks_polar = []

            for pxj in range(-self.jitter, self.jitter+1):
                for pyj in range(-self.jitter, self.jitter+1):
                    for ixj in range(-self.jitter, self.jitter+1):
                        for iyj in range(-self.jitter, self.jitter+1):
                            for mxj in range(-self.jitter, self.jitter+1):
                                for myj in range(-self.jitter, self.jitter+1):
                                    
                                    mask_jitter = mask.clone().detach()

                                    mask_jitter = torch.roll(mask_jitter, mxj, dims=3)

                                    if mxj > 0:
                                        mask_jitter[:,:,:,:mxj] = 0.0
                                    elif mxj < 0:
                                        mask_jitter[:,:,:,mxj:] = 0.0

                                    mask_jitter = torch.roll(mask_jitter, myj, dims=2)
                                    
                                    if myj > 0:
                                        mask_jitter[:,:,:myj,:] = 0.0
                                    elif myj < 0:
                                        mask_jitter[:,:,myj:,:] = 0.0
                                    
                                    pxyr_jitter = pupil_xyr.clone().detach()
                                    ixyr_jitter = iris_xyr.clone().detach()

                                    pxyr_jitter[:, 0] += pxj
                                    pxyr_jitter[:, 1] += pyj
                                    
                                    ixyr_jitter[:, 0] += ixj
                                    ixyr_jitter[:, 1] += iyj
                                    
                                    pxCirclePoints = (pxyr_jitter[:, 0].reshape(-1, 1) + pxyr_jitter[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device) #b x 512
                                    pyCirclePoints = (pxyr_jitter[:, 1].reshape(-1, 1) + pxyr_jitter[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device)  #b x 512
                                    
                                    ixCirclePoints = (ixyr_jitter[:, 0].reshape(-1, 1) + ixyr_jitter[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)).to(self.device)  #b x 512
                                    iyCirclePoints = (ixyr_jitter[:, 1].reshape(-1, 1) + ixyr_jitter[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)).to(self.device) #b x 512
                                    
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
                                    mask_polar = self.grid_sample(mask_jitter, grid_sample_mat, interp_mode='nearest') # always use nearest neighbor interpolation for mask
                                    mask_polar = (mask_polar>0.5).long() * 255
                                    
                                    images_polar.append(image_polar[0][0].cpu().numpy().astype(np.uint8))
                                    masks_polar.append(mask_polar[0][0].cpu().numpy().astype(np.uint8))

        return images_polar, masks_polar
    
    # (Fixed) Old implementation of Rubbersheet model-based Cartesian-to-polar transformation that uses nearest neighbor interpolation
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

    def extractCode(self, polar):
        with torch.no_grad():
            if polar is None:
                return None
            r = int(np.floor(self.filter_size / 2))
            polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0).to(self.device)
            #polar_t = (polar_t - polar_t.min()) / (polar_t.max() - polar_t.min())
            padded_polar = nn.functional.pad(polar_t, (r, r, 0, 0), mode='circular')
            codeContinuous = nn.functional.conv2d(padded_polar, self.torch_filter)
            codeBinary = torch.where(codeContinuous.squeeze(0) > 0, True, False).cpu().numpy()
            return codeBinary # The size of the code should be: 7 x (64 - filter_size) x 512 
    
    def extractMultipleCodes(self, polars):
        with torch.no_grad():
            polars_t = []
            for polar in polars:
                if polar is None:
                    print("One of the polar images is None. Unable to continue...")
                    return None
                r = int(np.floor(self.filter_size / 2))
                polar_t = torch.tensor(polar).float().unsqueeze(0).unsqueeze(0).to(self.device)
                polars_t.append(polar_t)
            polars_t = torch.cat(polars_t, 0)
            padded_polar = nn.functional.pad(polars_t, (r, r, 0, 0), mode='circular')
            codesContinuous = nn.functional.conv2d(padded_polar, self.torch_filter)
            codesBinary = torch.where(codesContinuous > 0, True, False).cpu().numpy()
            codes = []
            for i in range(codesBinary.shape[0]):
                codes.append(codesBinary[i]) # The size of the code should be: 7 x (64 - filter_size) x 512
        return codes

    def findMajorityVoteCode(self, codes, masks_polar):
        masked_codes = []
        masks_polar_binary = []
        r = int(np.floor(self.filter_size / 2))
        for code, mask_polar in zip(codes, masks_polar):
            mask_polar_binary = np.where(mask_polar > 127, True, False)
            masks_polar_binary.append(mask_polar_binary.astype(int))
            masked_codes.append(np.logical_and(code, mask_polar_binary[r:-r, :]).astype(int))
        masked_codes_sum = sum(masked_codes)
        masks_polar_sum = sum(masks_polar_binary)
        majorityVoteCode = np.where(masked_codes_sum > (masks_polar_sum[r:-r, :] * 0.5), True, False)
        mask_polar = np.where(masks_polar_sum > (len(masks_polar_binary)/2.0), 255, 0)
        return majorityVoteCode, mask_polar.astype(np.uint8)

    def matchCodes(self, code1, code2, mask1, mask2):
        r = int(np.floor(self.filter_size / 2))
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1[r:-r, :] > 127, True, False)
        mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
        if (np.sum(mask1_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits) or (np.sum(mask2_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits):
            print("Too small masks")
            return -1.0, -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(mask1_binary, np.roll(mask2_binary, xshift, axis=1))
            if np.sum(andMasks) == 0:
                scoreC.append(float('inf'))
            else:
                xorCodes = np.logical_xor(code1, np.roll(code2, xshift, axis=2))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.num_filters, 1, 1)))
                scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.num_filters))
            if self.score_norm:
                scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( np.sum(andMasks) / self.avg_num_bits )
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            print("Too small overlap between masks")
            return -1.0, -1.0
        scoreC_shift = scoreC_index - self.max_shift
        
        return scoreC_best, scoreC_shift
    
    def matchCodesJitter(self, codes1, code2, masks1, mask2):
        if self.multiproc:
            inputs = []
            for code1, mask1 in zip(codes1, masks1):
                inputs.append((self, code1, code2, mask1, mask2))
            with multiprocessing.Pool(self.num_workers) as p:
                scores = p.starmap(matchCodesGlobal, inputs)
            return np.min(np.array(scores))
        else:
            scores = []
            for code1, mask1 in zip(codes1, masks1):
                scores.append(self.matchCodes(code1, code2, mask1, mask2))
            return np.min(np.array(scores))

    def matchCodesShiftXY(self, code1, code2, mask1, mask2):
        r = int(np.floor(self.filter_size / 2))
        # Cutting off mask to (64-filter_size+1) x 512 and binarizing it.
        mask1_binary = np.where(mask1[r:-r, :] > 127, True, False) 
        mask2_binary = np.where(mask2[r:-r, :] > 127, True, False)
        if (np.sum(mask1_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits) or (np.sum(mask2_binary) <= self.threshold_frac_avg_bits * self.avg_num_bits):
            return -1.0, -1.0
        scoreC = []
        for xshift in range(-self.max_shift, self.max_shift+1):
            for ytrans in range(-int(round(self.max_shift/8)), int(round(self.max_shift/8))+1):
                if xshift != 0:
                    mask2_binary_shifted = np.roll(mask2_binary, xshift, axis=1)
                else:
                    mask2_binary_shifted = mask2_binary
                if ytrans != 0:
                    mask2_binary_shifted = np.roll(mask2_binary_shifted, ytrans, axis=0)
                    if ytrans > 0:
                        mask2_binary_shifted[:ytrans, :] = 0
                    elif ytrans < 0:
                        mask2_binary_shifted[ytrans:, :] = 0
                andMasks = np.logical_and(mask1_binary, mask2_binary_shifted)
                if np.sum(andMasks) == 0:
                    scoreC.append(float('inf'))
                else:
                    xorCodes = np.logical_xor(code1, np.roll(np.roll(code2, xshift, axis=2), ytrans, axis=1))
                    xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=0), (self.num_filters, 1, 1)))
                    scoreC.append(np.sum(xorCodesMasked) / (np.sum(andMasks) * self.num_filters))
                if self.score_norm:
                    scoreC[-1] = 0.5 - (0.5 - scoreC[-1]) * math.sqrt( np.sum(andMasks) / self.avg_num_bits )
        scoreC_index = np.argmin(np.array(scoreC))
        scoreC_best = scoreC[scoreC_index]
        if scoreC_best == float('inf'):
            return -1.0, -1.0
        
        shift_divide = int(round(self.max_shift/8)) * 2 + 1
        scoreC_shift = int(scoreC_index / shift_divide) - self.max_shift

        return scoreC_best, scoreC_shift
    
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
