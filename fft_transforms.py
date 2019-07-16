import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def jasnosc_kontrast(img) : 
    p = np.sum(img)
    
    width = img.shape[0]
    height = img.shape[1]
    
    J = p/(width*height)
    
    Kpom = np.sum(np.power(img-J,2))
    Kp = Kpom/(width*height)
    
    K = np.power(Kp,0.5)
    return J,K
#

orgimage = np.asarray(Image.open('Hough_okregi.png'))[:,:,0]

mask1 = np.zeros((orgimage.shape)) 
mask1[0:100,0:100] += 1
mask1 += np.fliplr(mask1)
mask1 += np.flipud(mask1)

mask2 = np.zeros((orgimage.shape))
mask2[0:80,50:80] += 1
mask2[50:80, 0:80] += 1
mask2 += np.fliplr(mask2)
mask2 += np.flipud(mask2)

j,k = jasnosc_kontrast(orgimage)

plt.figure("Fast Fourier's transform")
plt.subplot(3,4,1)
plt.title("Original")
plt.imshow(orgimage, cmap='gray')

plt.subplot(3,4,2)
str_after="After transform: "+str(round(j,2))+" /\ " + str(round(k,2))
plt.title(str_after)
plt.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(orgimage/255)))/255, cmap='gray',vmin=0,vmax=1)

plt.subplot(3,4,3)
after_transform = np.fft.fft2(orgimage/255)
after_transform = np.fft.fftshift(after_transform)
after_transform[np.where(after_transform == np.amax(after_transform))]/= 2
j,k = jasnosc_kontrast(after_transform)
str_after="After subdivision: "+str(round(j,2))+" /\ " + str(round(k,2))
plt.title(str_after)
plt.imshow(np.real(np.fft.ifft2(after_transform)),cmap='gray',vmin=0,vmax=1)

# np.fft.fftshift(widmo)
plt.subplot(3,4,5)
plt.imshow(mask1,cmap='gray',vmin=0,vmax=1)

after_transform[np.where(after_transform == np.amax(after_transform))]*=2
plt.subplot(3,4,6)
plt.imshow(np.real(np.fft.ifft2((after_transform*mask1))),cmap='gray',vmin=0,vmax=1)
plt.title("Mul with mask <-")

plt.subplot(3,4,7)
plt.imshow(mask2,cmap='gray',vmin=0,vmax=1)

plt.subplot(3,4,8)
plt.imshow(np.real(np.fft.ifft2(after_transform*(1 - mask1))),cmap='gray',vmin=0,vmax=1)
plt.title("Mul with mask <-")

plt.subplot(3,4,9)
plt.imshow(1 - (mask1),cmap='gray',vmin=0,vmax=1)

plt.subplot(3,4,10)
plt.imshow(np.real(np.fft.ifft2(after_transform*mask2)),cmap='gray',vmin=0,vmax=1)
plt.title("Mul with mask <-")

plt.subplot(3,4,11)
plt.imshow(1 - (mask2),cmap='gray',vmin=0,vmax=1)

plt.subplot(3,4,12)
plt.imshow(np.real(np.fft.ifft2(after_transform*(1 - mask2))),cmap='gray',vmin=0,vmax=1)
plt.title("Mul with mask <-")

plt.show()