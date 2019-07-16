#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
#from scipy import misc as mc

#Loading an image and referencing it to a variable
#mc.imread needs pillow (Python Image Loading) but it is already depreciated
#instead, please use: 
#    1. import imageio
#     * imageVar = imageio.imread(...)
#    2. from matplotplib.pyplot import imread

imageVar = imread('Hough_okregi.png')

#float32 - floating point type (how to change to int?)
print(imageVar.dtype)
#firstly - size of image - 256 x 256, third arg number of layers [ RED GREEN BLUE APLHA ] 
print(imageVar.shape)


#printing original image
plt.figure("Obrazek w oryginalnych kolorach - tytul glowny")
plt.imshow(imageVar) # Passing image to drawing function
plt.title("Obrazek w oryginalnych kolorach - podtytul")
plt.draw() # warn: plt.show() doesn't like waitforbuttonpress

plt.waitforbuttonpress(0)
plt.close()

# # # # 
plt.figure("Rozne kombinacje kolorow")

# One layer of image
plt.subplot(2,2,1)
layernumber = 2
onelayer_print = "one layer of image: (" + str(layernumber) + ")"
plt.title(onelayer_print)
plt.imshow(imageVar[:,:,layernumber])

# One layer of image in grayscale
plt.subplot(2,2,2)
plt.title("image in grayscale")
plt.imshow( imageVar[:,:,0], cmap='gray' ) #vmin = 0, vmax = 255 ?  

# Converting image to grayscale
imagegray1 = 0.299 * imageVar[:,:,0] + 0.587 * imageVar[:,:,1] + 0.114 * imageVar[:,:,2] 
imagegray2 = 0.2126 * imageVar[:,:,0] + 0.7152 * imageVar[:,:,2] + 0.0722 * imageVar[:,:,2]
# - - - - - - - - - - - - - - - 

# Image in grayscale1
plt.subplot(2,2,3)
grayscale1_print = "image in grayscale1: (" + str(0.299) + ")" + "(" + str(0.587) + ")" + "(" + str(0.114) + ")"
plt.title(grayscale1_print)
plt.imshow(imagegray1, cmap='gray')

# Image in grayscale2
plt.subplot(2,2,4)
grayscale1_print2 = "image in grayscale2: (" + str(0.2126) + ")" + "(" + str(0.7152) + ")" + "(" + str(0.0722) + ")"
plt.title(grayscale1_print2)
plt.imshow(imagegray2, cmap='gray')
plt.draw()

plt.waitforbuttonpress(0)
plt.close()
# # # 