import sys
from matplotlib.colors import to_rgba_array

from scipy.linalg import norm
from scipy import sum, average,ndimage
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2

THRESHOLD_BINARIZE=0.6

def convert_grayscale(img):
    if len(img.shape)==3:
        return np.average(img,-1)
    else:
        return img

def binerize(img,threshold):
    newImg=img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i,j]>threshold):
                newImg[i,j]=1
            else:
                newImg[i,j]=0
    return newImg

def main(folderRef,folderTest): 
    #Get alls files
    dirsRef=os.listdir(folderRef)
    dirsTest=os.listdir(folderTest)

    #boucle de traitement entre chaque image de référence et image de test
    for i in range(len(dirsRef)): 
        fileRef=folderRef+"/"+dirsRef[i]
        imgRef=plt.imread(fileRef)
       

        fileTest=folderTest+"/"+dirsTest[i]
        imgTest=plt.imread(fileTest)

        #convertion en nuance de gris
        imgRef=convert_grayscale(imgRef)
        imgTest=convert_grayscale(imgTest)

        #Binearisation
        imgRefBinary=binerize(imgRef,THRESHOLD_BINARIZE) 
        imgTestBinary=binerize(imgTest,THRESHOLD_BINARIZE)
         
        plt.imshow(imgRefBinary,cmap='gray')     
        plt.show()
        sobelxyRef = cv2.Sobel(src=imgRefBinary, ddepth=cv2.CV_64F, dx=1, dy=1,ksize=5)
        sobelxyTest = cv2.Sobel(src=imgTestBinary, ddepth=cv2.CV_64F, dx=1, dy=1,ksize=5)

        sobelxyRef=binerize(sobelxyRef,THRESHOLD_BINARIZE) 
        sobelxyTest=binerize(sobelxyTest,THRESHOLD_BINARIZE)
        plt.subplot(1,2,1)
        plt.imshow(sobelxyRef,cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(sobelxyTest,cmap='gray')
        plt.show()
        
main("test","test2")