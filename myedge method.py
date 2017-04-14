import cv2.cv as cv
import sys
import cv
from opencv.cv import *
from opencv.highgui import *
import cv2
import matplotlib.pyplot as plt
from pil import Image
import scipy
import scipy.misc


ll='/Users/yuelingqin/Desktop/infor/4300image/'+'44.jpeg'
im=cv.LoadImage(ll);

# Laplace on a gray scale picture
gray = cv.CreateImage(cv.GetSize(im), 8, 1)
cv.CvtColor(im, gray, cv.CV_BGR2GRAY)

aperture=3

dst = cv.CreateImage(cv.GetSize(gray), cv.IPL_DEPTH_32F, 1)
cv.Laplace(gray, dst,aperture)

cv.Convert(dst,gray)

thresholded = cv.CloneImage(im)
cv.Threshold(im, thresholded, 50, 255, cv.CV_THRESH_BINARY_INV)

cv.ShowImage('Laplaced grayscale',gray)
#------------------------------------

# Laplace on color
planes = [cv.CreateImage(cv.GetSize(im), 8, 1) for i in range(3)]
laplace = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 1)
colorlaplace = cv.CreateImage(cv.GetSize(im), 8, 3)

cv.Split(im, planes[0], planes[1], planes[2], None) #Split channels to apply laplace on each
for plane in planes:
    cv.Laplace(plane, laplace, 3)
    cv.ConvertScaleAbs(laplace, plane, 1, 0)

cv.Merge(planes[0], planes[1], planes[2], None, colorlaplace)

cv.ShowImage('Laplace Color', colorlaplace)

cv2.waitKey(33)
    # if k==27:    # Esc key to stop
    #     break
    # elif k==-1:  # normally -1 returned,so don't print it
    #     continue

# import scipy
# scipy.misc.imsave(ll+'colorlaplace.jpg', colorlaplace)

#scipy.misc.toimage(colorlaplace,  cmin=0.0, cmax=255).save('colorlaplace.jpg')
# im = Image.fromarray(colorlaplace)
# im.save("colorlaplace.jpg")








###--------
import cv2.cv as cv
import sys
import cv
from opencv.cv import *
from opencv.highgui import *
ll='/Users/yuelingqin/Desktop/infor/4300image/'+'44.jpeg'


im=cv.LoadImage(ll);
im = cv.CvtColor(im, cv2.COLOR_BGR2GRAY)

dst_32f = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_32F, 1)

neighbourhood = 3
aperture = 3
k = 0.01
maxStrength = 0.0
threshold = 0.01
nonMaxSize = 3

cv.CornerHarris(im, dst_32f, neighbourhood, aperture, k)

minv, maxv, minl, maxl = cv.MinMaxLoc(dst_32f)

dilated = cv.CloneImage(dst_32f)
cv.Dilate(dst_32f, dilated) # By this way we are sure that pixel with local max value will not be changed, and all the others will

localMax = cv.CreateMat(dst_32f.height, dst_32f.width, cv.CV_8U)
cv.Cmp(dst_32f, dilated, localMax, cv.CV_CMP_EQ) #compare allow to keep only non modified pixel which are local maximum values which are corners.

threshold = 0.01 * maxv
cv.Threshold(dst_32f, dst_32f, threshold, 255, cv.CV_THRESH_BINARY)

cornerMap = cv.CreateMat(dst_32f.height, dst_32f.width, cv.CV_8U)
cv.Convert(dst_32f, cornerMap) #Convert to make the and
cv.And(cornerMap, localMax, cornerMap) #Delete all modified pixels

radius = 3
thickness = 2

l = []
for x in range(cornerMap.height): #Create the list of point take all pixel that are not 0 (so not black)
    for y in range(cornerMap.width):
        if cornerMap[x,y]:
            l.append((y,x))

for center in l:
    cv.Circle(im, center, radius, (255,255,255), thickness)


cv.ShowImage("Image", im)
cv.ShowImage("CornerHarris Result", dst_32f)
cv.ShowImage("Unique Points after Dilatation/CMP/And", cornerMap)

cv.WaitKey(33)




# #------


#Get edges
morphed = cv.CloneImage(im)
cv.MorphologyEx(im, morphed, None, None, cv.CV_MOP_GRADIENT) # Apply a dilate - Erode

cv.Threshold(morphed, morphed, 30, 255, cv.CV_THRESH_BINARY_INV)

cv.ShowImage("Image", im)
cv.ShowImage("Morphed", morphed)

#--------
pi = math.pi #Pi value

dst = cv.CreateImage(cv.GetSize(im), 8, 1)

cv.Canny(im, dst, 200, 200)
cv.Threshold(dst, dst, 100, 255, cv.CV_THRESH_BINARY)

#---- Standard ----
color_dst_standard = cv.CreateImage(cv.GetSize(im), 8, 3)
cv.CvtColor(im, color_dst_standard, cv.CV_GRAY2BGR)#Create output image in RGB to put red lines

lines = cv.HoughLines2(dst, cv.CreateMemStorage(0), cv.CV_HOUGH_STANDARD, 1, pi / 180, 100, 0, 0)
for (rho, theta) in lines[:100]:
    a = math.cos(theta) #Calculate orientation in order to print them
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
    pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
    cv.Line(color_dst_standard, pt1, pt2, cv.CV_RGB(255, 0, 0), 2, 4) #Draw the line

#---- Probabilistic ----
color_dst_proba = cv.CreateImage(cv.GetSize(im), 8, 3)
cv.CvtColor(im, color_dst_proba, cv.CV_GRAY2BGR) # idem

rho=1
theta=pi/180
thresh = 50
minLength= 120 # Values can be changed approximately to fit your image edges
maxGap= 20

lines = cv.HoughLines2(dst, cv.CreateMemStorage(0), cv.CV_HOUGH_PROBABILISTIC, rho, theta, thresh, minLength, maxGap)
for line in lines:
    cv.Line(color_dst_proba, line[0], line[1], cv.CV_RGB(255, 0, 0), 2, 8)

cv.ShowImage('Image',im)
cv.ShowImage("Cannied", dst)
cv.ShowImage("Hough Standard", color_dst_standard)
cv.ShowImage("Hough Probabilistic", color_dst_proba)





# #-----3234232152435-------------------------------

# orig = im
# im = cv.CreateImage(cv.GetSize(orig), 8, 1)
# cv.CvtColor(orig, im, cv.CV_BGR2GRAY)
# #Keep the original in colour to draw contours in the end

# cv.Threshold(im, im, 128, 255, cv.CV_THRESH_BINARY)
# cv.ShowImage("Threshold 1", im)

# element = cv.CreateStructuringElementEx(5*2+1, 5*2+1, 5, 5, cv.CV_SHAPE_RECT)

# cv.MorphologyEx(im, im, None, element, cv.CV_MOP_OPEN) #Open and close to make appear contours
# cv.MorphologyEx(im, im, None, element, cv.CV_MOP_CLOSE)
# cv.Threshold(im, im, 128, 255, cv.CV_THRESH_BINARY_INV)
# cv.ShowImage("After MorphologyEx", im)
# # --------------------------------

# vals = cv.CloneImage(im) #Make a clone because FindContours can modify the image
# contours=cv.FindContours(vals, cv.CreateMemStorage(0), cv.CV_RETR_LIST, cv.CV_CHAIN_APPROX_SIMPLE, (0,0))

# _red = (0, 0, 255); #Red for external contours
# _green = (0, 255, 0);# Gren internal contours
# levels=2 #1 contours drawn, 2 internal contours as well, 3 ...
# cv.DrawContours (orig, contours, _red, _green, levels, 2, cv.CV_FILLED) #Draw contours on the colour image

# cv.ShowImage("Image", orig)
























