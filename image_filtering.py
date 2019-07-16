import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Numpy Array consists of two elements: data buffer and the view
# https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

#FILTR ROBERTSA
maskR1_V=np.matrix('-1,0;1,0');
maskR1_H=np.matrix('-1,1;0,0');
maskR2_V=np.matrix('0,1;-1,0');
maskR2_H=np.matrix('1,0;0,-1');

#FILTR LAPLACE'A
maskL=np.matrix('0,1,0;1,-4,1;0,1,0');
maskL1=np.matrix('1,1,1;1,-8,1;1,1,1');

#FILTR PREWITTA
maskP=np.matrix('-1,-1,0;-1,0,1;0,1,1');
maskP1=np.matrix('-1,-1,-1;0,0,0;1,1,1');
maskP2=np.matrix('0,1,1;-1,0,1;-1,-1,0');
maskP3=np.matrix('-1,-1,0;-1,0,0;0,1,1');

#FILTR SOBELA
maskS=np.matrix('-1,0,1;-2,0,2;-1,0,1');
maskS1=np.matrix('-1,-2,-1;0,0,0;1,2,1');
maskS2=np.matrix('0,1,2;-1,0,1;-2,-1,0');
maskS3=np.matrix('-2,-1,0;-1,0,1;0,1,2');

#FILTR KIRSCHA
maskK=np.matrix('-3,-3,5;-3,0,5;-3,-3,5');
maskK1=np.matrix('-3,5,5;-3,0,5;-3,-3,-3');
maskK2=np.matrix('5,5,5;-3,0,-3;-3,-3,-3');
maskK3=np.matrix('5,5,-3;5,0,-3;-3,-3,-3');

maskK4=np.matrix('5,-3,-3;5,0,-3;5,-3,-3');
maskK5=np.matrix('-3,-3,-3;5,0,-3;5,5,-3');
maskK6=np.matrix('-3,-3,-3;-3,0,-3;5,5,5');
maskK7=np.matrix('-3,-3,-3;-3,0,5;-3,5,5');

######################################################################
def ffilter(img, mask):
     
    #Down, upper row offset from mask center
    offset_r2 = int(mask.shape[0]) - int(mask.shape[0]/2) - 1
    offset_r1 = int(mask.shape[0]) - offset_r2 - 1 

    #Right, left column (...)
    offset_c2 = int(mask.shape[1]) - int(mask.shape[1]/2) - 1
    offset_c1 = int(mask.shape[1]) - offset_c2 - 1
    
    #Creating matrices for mirroring - Left Side, RightSide, UpSide, DownSide
    LS = np.ones( ((offset_r1 + img.shape[0] + offset_r2),offset_c1) )
    RS = np.ones( ((offset_r1 + img.shape[0] + offset_r2),offset_c2) )
    US = np.ones( (offset_r1,img.shape[1]) )
    DS = np.ones( (offset_r2,img.shape[1]) )
    
    #Mirroring upside and downside
    US[::-1, :] = img[:US.shape[0]:1,:]
    DS[::-1, :] = img[ (img.shape[0] - offset_r2):img.shape[0]:1,:]
    
    #Concatenating up - original image - down
    POW = np.concatenate( (US,img,DS), axis=0)
    
    #Mirroring left and right side
    LS[:, ::-1] = POW[:, :offset_c1:1]
    RS[:, ::-1] = POW[:, (img.shape[1] - offset_c2):img.shape[1]:1]
    
    #Concatenating left - original - right side
    POW = np.concatenate( (LS,POW,RS), axis=1)
    
    #Calculate norm of the mask, and copy image
    norm = 1 if np.sum(mask) == 0 else np.sum(mask)
    COPYPOW = np.copy(POW)
    
    #Filtering operation
    for i,j in np.ndindex(img.shape):
        p = np.copy(POW[i:mask.shape[0]+i,j:mask.shape[1]+j])
        p[offset_r1,offset_c1] = COPYPOW[offset_c1 + i, offset_r1+j] = np.sum(np.multiply(p,mask))/norm 
    
    #return filtered image without mirror parts
    return COPYPOW[offset_r1:offset_r1+img.shape[0],offset_c1:offset_c1+img.shape[1]]
#


# # # MAIN:
image_to_filter = np.asarray(Image.open('litery_1.png'))[:,:,0]

plt.figure("Image filtering")
plt.subplot(2,3,1)
plt.title("Roberts ")
plt.imshow( np.sqrt((np.power(ffilter(image_to_filter, maskR2_V),2) + np.power(ffilter(image_to_filter,maskR2_H),2) ) ), cmap='gray', vmin=0, vmax=255)
#  
plt.subplot(2,3,2)
plt.title("Laplace ")
# ffilter(ffilter(image_to_filter, maskL),maskL1)
plt.imshow( np.sqrt((np.power(ffilter(image_to_filter, maskL),2)+np.power(ffilter(image_to_filter,maskL1),2) ) ), cmap='gray', vmin=0, vmax=255)
  
plt.subplot(2,3,3)
plt.title("Prewitt ")
# ffilter(ffilter(image_to_filter, maskP),maskP2)
plt.imshow( np.sqrt((np.power(ffilter(image_to_filter, maskP),2)+np.power(ffilter(image_to_filter,maskP2),2) ) ), cmap='gray', vmin=0, vmax=255)
  
plt.subplot(2,3,4)
plt.title("Sobel ")
# ffilter(ffilter(image_to_filter, maskS),maskS2)
plt.imshow( np.sqrt((np.power(ffilter(image_to_filter, maskS),2)+np.power(ffilter(image_to_filter,maskS2),2) ) ), cmap='gray', vmin=0, vmax=255)
  
plt.subplot(2,3,5)
plt.title("Kirsh ")
# ffilter(ffilter(image_to_filter, maskK),maskK1)
plt.imshow( np.sqrt((np.power(ffilter(image_to_filter, maskK),2)+np.power(ffilter(image_to_filter,maskK1),2) ) ), cmap='gray', vmin=0, vmax=255)
  
plt.subplot(2,3,6)
plt.title("Original image ")
plt.imshow(image_to_filter,cmap='gray')
plt.show()