import cv2
import numpy as np
import math

from picamera import PiCamera
from time import sleep

#camera = PiCamera()
#camera.resolution = (640,480)
#camera.capture('/home/pi/Desktop/image0.jpg')
img = cv2.imread('/home/pi/Desktop/image0.jpg', 1)
img = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,15,29)
kernel = np.ones((2,2),np.uint8)

closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,iterations=1)
dilation = cv2.dilate(closing, kernel, iterations=7)
lines = cv2.HoughLinesP(closing,0.25,np.pi/200, 15,np.array([]), 17,3)

a = len(lines)

h, w = img.shape[:2]
mask = np.zeros((h+2,w+2), np.uint8)

coeff = []
for i in range(a):
    [x1,y1,x2,y2] = lines[i][0]
    lenght = int(math.sqrt((x1-x2)**2+(y1-y2)**2))
    if x1 != x2:
        k = round((float(y2 - y1))/(float(x2 - x1)),3)
        b = y1 - k*x1
    else:
        k = -337
        b = y1 - k*x1
    coeff.append([k,b,lenght,(x1+x2)/2,(y1+y2)/2])
coeff = np.array(coeff)

def parallel(line_i, a, val):
    maxDist = 0
    for i in range(0,a):
        if (abs((math.atan(coeff[i][0])-math.atan(coeff[line_i][0]))) < 0.1) and ( \
            i!=line_i) and (lines[i][0][0]!=0) and (val == dilation[lines[i][0][1],lines[i][0][0]]):
            dist = int((abs(coeff[i][1]-coeff[line_i][1]))/(math.sqrt(1+math.pow(coeff[i][0],2))))
            if dist > maxDist:
                maxDist = dist    
                best_I = i
        
    return best_I

pairs = np.zeros((16,3),np.uint16)
best_I, best_J, maxDist = [0,0,0]
k=0
###   removin' close lines   ###
for i in range(0,a):
    for j in range(0,a):
        cdst = math.sqrt(math.pow(coeff[j][3]-coeff[i][3],2) + math.pow(coeff[j][4]-coeff[i][4],2))
        if (abs((math.atan(coeff[i][0])-math.atan(coeff[j][0]))) < 0.2) and (cdst < 100):
            dist = int((abs(coeff[i][1]-coeff[j][1]))/(math.sqrt(1+math.pow(coeff[i][0],2))))
            if (coeff[j][2] < coeff[i][2]) and (dist<6):
                lines[j]=0
                k = k+1
################################

###  showin' remaining lines ###
for i in range(a):
    cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), \
             (0,255,0),1,cv2.LINE_AA)
################################

### findin' parallel line   ####
m=1

a = len(lines)

for i in range(a):
    best_J=-1
    maxDist = 0
    if (lines[i][0][2] > pairs[(dilation[lines[i][0][1],lines[i][0][0]]-100)/10][2]) and (lines[i][0][0]!=0):
        for j in range(a):
            if (abs((math.atan(coeff[i][0])-math.atan(coeff[j][0]))) < 0.1) and (i!=j) and ( \
                lines[j][0][0]!=0):
                if (dilation[lines[j][0][1],lines[j][0][0]] == 255):
                    cv2.floodFill(dilation,mask,(lines[j][0][0],lines[j][0][1]),m*10+100)
                    m=m+1
                if (dilation[lines[j][0][1],lines[j][0][0]] == dilation[lines[i][0][1],lines[i][0][0]]):
                    dist = int((abs(coeff[j][1]-coeff[i][1]))/(math.sqrt(1+math.pow(coeff[j][0],2))))
                    if (dist >= maxDist):
                        maxDist = dist              
                        best_J = j
    
    if (best_J != -1):
        d = dilation[lines[best_J][0][1],lines[best_J][0][0]]
        best_I = parallel(best_J, a, d)
        pairs[(d-100)/10] = [best_I,best_J,(lines[best_J][0][2] + lines[best_I][0][2])/2]
        

################################
for i in range(1,15):
    if (pairs[i][2] != 0):
        cv2.line(img, (lines[pairs[i][0]][0][0], lines[pairs[i][0]][0][1]), \
                 (lines[pairs[i][0]][0][2], lines[pairs[i][0]][0][3]), \
                 (0,0,255),3,cv2.LINE_AA)
        cv2.line(img, (lines[pairs[i][1]][0][0], lines[pairs[i][1]][0][1]), \
                 (lines[pairs[i][1]][0][2], lines[pairs[i][1]][0][3]), \
                 (0,0,255),3,cv2.LINE_AA)
print('Green lines removed: %d'% k)
print('Objects detected: %d'% m)
cv2.imshow('img',img)
#cv2.imshow('edges',np.hstack([edges]))
cv2.imshow('dilation',dilation)
cv2.waitKey(000)
