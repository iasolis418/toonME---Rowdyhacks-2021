# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:03:53 2021

@author: Aaron
"""

import cv2 as cv
from sklearn.cluster import MiniBatchKMeans

#CHALK STYLE
def chalk(img_path):
    img = cv.imread(img_path)
    lower_res = img.copy()
 
    #Create a Gaussian Pyramid
    gaussian_pyramid = [lower_res]
    for i in range(2):
        lower_res = cv.pyrDown(lower_res)
        gaussian_pyramid.append(lower_res)

    #Use images from pyramid to create laplacian image
    size = (gaussian_pyramid[0].shape[1], gaussian_pyramid[0].shape[0])
    higher_res = cv.pyrUp(gaussian_pyramid[1], dstsize=size)
    laplacian = cv.subtract(gaussian_pyramid[0], higher_res)

    #Adjust laplacian to look like chalk image
    gray = cv.cvtColor(laplacian, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (1,1), 0)
    ret, chalk = cv.threshold(blur, 3.5, 255, cv.THRESH_BINARY)

    return chalk

#COMIC STYLE
def cel(img_path):
  img = cv.imread(img_path)
  (h, w) = img.shape[:2]
  
  #block size ddtermined by size of image
  blkSize = (int)(7 * (h + w) / 840)
  blkSize = blkSize + blkSize % 2 - 1

  c = (int)(blkSize / 5) + 1
  num_down = (int)(c / 2)
  num_bilateral = (int)(blkSize / 2)


  img_color = cv.cvtColor(img, cv.COLOR_BGR2LAB)

  #scales the image down
  for x in range(num_down):
    img_color = cv.pyrDown(img_color)
  
  #bilateral filter smooths out colors
  for x in range(num_bilateral):
    img_color = cv.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
  
  #scale back up
  for x in range(num_down):
    img_color = cv.pyrUp(img_color)


  (h, w) = img_color.shape[:2]
  
  img_color = img_color.reshape((img_color.shape[0] * img_color.shape[1], 3))

  #K-Means clustering reudces number of colors on screen
  clt = MiniBatchKMeans(n_clusters=25)
  labels = clt.fit_predict(img_color)
  quant = clt.cluster_centers_.astype("uint8")[labels]

  quant = quant.reshape((h, w, 3))
  img_color = img_color.reshape((h, w, 3))

  quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
  img_color = cv.cvtColor(img_color, cv.COLOR_LAB2BGR)

  #lines where between different colors
  img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
  img_blur = cv.medianBlur(img_gray, 5)

  #threshold determines size of line
  img_edge = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=blkSize, C=c)
  img_edge = cv.cvtColor(img_edge, cv.COLOR_GRAY2BGR)

  #apply edges
  img_w_edges = cv.bitwise_and(img_color, img_edge)
  
  return img_w_edges

#DRIVER
if __name__ == "__main__":
    img_path = input("Hello! Please provide image path: ")
    menu_choice = input("Please enter 1 for chalk style, or 2 for Comic style: ")
    print("Thank you for your time! Press 'ESC' to close image window.")
    
    if(menu_choice == "1"):
        img = chalk(img_path)
    
    else:
        img = cel(img_path)
        
    cv.imshow('Result', img)
    
    if cv.waitKey(1) == 27:
        cv.destroyAllWindows()