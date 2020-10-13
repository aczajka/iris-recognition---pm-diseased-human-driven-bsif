function [IMAGE_POL,MASK_POL] = pmIrisCartesianToPolar(IMAGE,MASK,pupilData,irisData)

POL_H = 64;
POL_W = length(pupilData.theta);

IMAGE_POL = uint8(zeros(POL_H,POL_W));
MASK_POL = uint8(zeros(POL_H,POL_W));

for j = 1:POL_W
    
    for i = 1:POL_H
        
        radius = i / POL_H;
        x = round((1-radius) * pupilData.xCirclePoints(j) + radius * irisData.xCirclePoints(j));
        y = round((1-radius) * pupilData.yCirclePoints(j) + radius * irisData.yCirclePoints(j));
        
        IMAGE_POL(i,j) = IMAGE(y,x);
        MASK_POL(i,j) = 255*MASK(y,x);
    end
end