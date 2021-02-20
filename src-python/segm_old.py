import numpy as np
import torch
import time
import collections as col
from statistics import mean 

from PIL import Image
from argparse import ArgumentParser
import torch.nn as nn

from torch.autograd import Variable
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor

from modules.network import UNet
from modules.transform import Relabel, ToLabel

import cv2
font = cv2.FONT_HERSHEY_SIMPLEX 

# CCNet stuff:
NUM_CHANNELS = 1
NUM_CLASSES = 2
INPUT_SIZE = (320,240)
input_transform = Compose([
    ToTensor(),
])

# Misc
scaledImageH = 510
eyeMarkerSize = 10

def increase_brightness(img, value=30):
    lim = 255 - value
    img[img > lim] = 255
    img[img <= lim] += value
    return img


def getEyeImages(img,faceCascade,eyeCascade):

    res_scale = scaledImageH / img.shape[0]
    imgScaled = cv2.resize(img, (int(scaledImageH/0.5625),scaledImageH))

    eyeCroppedColorR = []
    eyeCroppedColorL = []
    faceBB = (0,0,0,0)
    eyeLBB = (0,0,0,0)
    eyeRBB = (0,0,0,0)
    
    faces = faceCascade.detectMultiScale(
        imgScaled,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (64, 64),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if np.array(faces).size > 0:

        (x1, y1, w1, h1) = faces[0]
        imgScaled = cv2.rectangle(imgScaled, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 1)

        x = int(x1/res_scale)
        y = int(y1/res_scale)
        w = int(w1/res_scale)
        h = int(h1/res_scale)
        faceBB = (x,y,w,h)
        

        # Right eye
        faceROI = img[y:y+int(h/2), x:x+int(w/2)]
        eye = eyeCascade.detectMultiScale(faceROI)

        if np.array(eye).size > 0:
            eyeRBB = eye[0] # (ex,ey,ew,eh)
            cx = eyeRBB[0] + int(eyeRBB[2]/2)
            cy = eyeRBB[1] + int(eyeRBB[3]/2)
            
            eyeCroppedColorR = faceROI[cy-int(0.75*eyeRBB[2]/2):cy+int(0.75*eyeRBB[2]/2), eyeRBB[0]:eyeRBB[0]+eyeRBB[2]]
            imgScaled = cv2.circle(imgScaled, (int((x+cx)*res_scale), int((y+cy)*res_scale)), eyeMarkerSize, (0, 255, 0), 1)
            
        # Left eye
        faceROI = img[y:y+int(h/2), x+int(w/2):x+w]
        eye = eyeCascade.detectMultiScale(faceROI)

        if np.array(eye).size > 0:
            eyeLBB = eye[0] # (ex,ey,ew,eh)
            cx = eyeLBB[0] + int(eyeLBB[2]/2)
            cy = eyeLBB[1] + int(eyeLBB[3]/2)
            
            eyeCroppedColorL = faceROI[cy-int(0.75*eyeLBB[2]/2):cy+int(0.75*eyeLBB[2]/2), eyeLBB[0]:eyeLBB[0]+eyeLBB[2]]
            imgScaled = cv2.circle(imgScaled, (int((x+cx+w/2)*res_scale), int((y+cy)*res_scale)), eyeMarkerSize, (0, 255, 0), 1)

    return eyeCroppedColorR, eyeCroppedColorL, imgScaled


def getPupilRadius(eyeCropped,model,softmax):
    pupilR = -1.0

    if np.array(eyeCropped).size > 0: 
        eyeCroppedGray = cv2.cvtColor(eyeCropped, cv2.COLOR_BGR2GRAY)

        eyeCroppedGray = cv2.resize(eyeCroppedGray, INPUT_SIZE)
        eyeCropped = cv2.resize(eyeCropped, INPUT_SIZE)

        outputs = model(Variable(input_transform(eyeCroppedGray).unsqueeze(0)))
        logprob = softmax(outputs).data.cpu().numpy()
        pred = np.argmax(logprob, axis=1)*255
        pred = Image.fromarray(pred[0].astype(np.uint8))
        pred = np.array(pred)

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

        eyeCropped[:,:,2] = pred

        # Hough transform
        circles = cv2.HoughCircles(pred,cv2.HOUGH_GRADIENT,1,20,param1=10,param2=10,minRadius=8,maxRadius=32)
            
        if circles is not None:
            c = np.around(circles[0][0])
            pupilR = c[2]
            eyeCropped = cv2.circle(eyeCropped,(c[0],c[1]),pupilR,(0,255,0),2)

    return pupilR, eyeCropped

def main(args):

    # Moving average stuff:
    ma_window = 25 # ~ 1 second for FPS=25
    ma_registerR = col.deque([],ma_window)
    ma_registerL = col.deque([],ma_window)

    prev_pupilR = 0
    prev_pupilL = 0

    # Webcam init 
    cam = cv2.VideoCapture(int(args.webcam))

    # Face and eye detection Haar cascades
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	
    # CCNet model loading (iris segmentation)
    model = UNet(NUM_CLASSES, NUM_CHANNELS)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    if args.state:
        try:
            if args.cuda:
                model.load_state_dict(torch.load(args.state))
            else:
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))
                # print("model state loaded")
        except AssertionError:
            print("assertion error")
            model.load_state_dict(torch.load(args.state,
                map_location = lambda storage, loc: storage))
    model.eval()
    softmax = nn.LogSoftmax(dim=1)

    # Main loop
    while (True):
        retval, img = cam.read()

        # Get cropped left and right eye images
        eyeCroppedColorR, eyeCroppedColorL, imgScaled = getEyeImages(img,faceCascade,eyeCascade)

        # Estimate the pupil radii (in pixels; relative to the original webcam image resolution)
        pupilR, eyeCroppedColorRsegm = getPupilRadius(eyeCroppedColorR,model,softmax)
        pupilL, eyeCroppedColorLsegm = getPupilRadius(eyeCroppedColorL,model,softmax)

        # Do some stuff with these radii, e.g., smooth them using moving-window average:
        if pupilR > 0:
            ma_registerR.append(pupilR)
            prev_pupilR = pupilR
        else:
            ma_registerR.append(prev_pupilR)
        pupilR_size_ma = mean(ma_registerR)

        if pupilL > 0:
            ma_registerL.append(pupilL)
            prev_pupilL = pupilL
        else:
            ma_registerL.append(prev_pupilL)
        pupilL_size_ma = mean(ma_registerL)

        # Embed results into the video preview
        if np.array(eyeCroppedColorR).size > 0: 
            imgScaled[10:250,10:330] = eyeCroppedColorRsegm
            imgScaled = cv2.putText(imgScaled, "right eye r={:02.1f} px".format(pupilR_size_ma), (20,30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if np.array(eyeCroppedColorL).size > 0: 
            imgScaled[260:500,10:330] = eyeCroppedColorLsegm
            imgScaled = cv2.putText(imgScaled, "left eye r={:02.1f} px".format(pupilL_size_ma), (20,280), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Pupilometer", imgScaled)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--state')
    parser.add_argument('--eye')
    parser.add_argument('--webcam')

    main(parser.parse_args())

