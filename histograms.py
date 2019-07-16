import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def histogram (img, N):
    
    histvec = np.zeros((N-1))
    
    tempsr = np.linspace(0,255,N)
    srvec = (tempsr[:-1]+tempsr[1:])/2
    
    for i in range(N-1):
        histvec[i] = np.sum((img > tempsr[i]) & (img <= tempsr[i+1]))
            
    return histvec, srvec
#

def linear_span (imgorg,x1,x2):
    
    img = imgorg.copy()
    
    img[img <= x1] = 0
    img[img > x2] = 255

    return img
#

def non_linear_span (imgorg,x1,x2,s):
    
    img = imgorg.copy()
    fx = np.linspace(x1,x2,100)
    
    img[img <= x1] = 0
    img[img > x2] = 255

    
    if s == "power":
        img[img >= x1] = np.power(img[img >= x1], 2)
        fy = np.power(fx,2)

    elif s == "sqrt":
        img[img >= x1] = np.sqrt(img[img >= x1])
        fy = np.sqrt(fx)

    elif s == "log":
        img[img >= x1] = np.log10(img[img >= x1])
        fy = np.log10(fx)

    elif s == "log_power":
        img[img >= x1] = np.power(5, img[img >= x1])
        fy = 2 ** fx
    else:
        print("Incorrect command 's' ")

    return img,fy
#

def equalization_hist (imgorg,hist):
    
    D = np.zeros(hist.shape[0]) #tworzymy dystrybuante
    img = imgorg.copy()

    # wypelniamy dystrybuante
    for i in range (0,hist.shape[0]):
        D[i]=np.sum(hist[0:i])

    D2 = np.zeros(hist.shape[0])

    dfmin = np.where(D > 0)
    
    #Wyrownywanie histogramu wg wzoru
    for i in range (0,hist.shape[0]):
        D2[i]=np.round(((D[i]-dfmin[0][0])/((img.shape[0]*img.shape[1])-dfmin[0][0]))*255) #dfmin[0][0]- najmniejszy element
    
    #Wyrownywanie obrazu
    for i in range(0,hist.shape[0]):
        img[img == i] = D2[i]

    return D2, img
#

#MAIN:

image = np.asarray(Image.open('kierowca.png'))[:,:,0]
#Ilosc przedzialow: 
N = 100

#Rozciagniecie liniowe:
x1 = 50
x2 = 120

plt.figure("Histogram operations")

#FIRST IMAGE: hist & img # # # 
H,sr = histogram(image, N)
HP = H.copy()/np.sum(H)
plt.subplot(3, 4, 1)
plt.title("Histogram org: "+str(N))
plt.bar(sr, HP, 100/N)

plt.subplot(3, 4, 9)
plt.title("Original: ")
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# # # 

#LINEAR SPAN: hist & img # # #
plt.subplot(3,4,2)
spannedlinear = linear_span(image, x1, x2)
H,sr = histogram(spannedlinear, N)
HP = H.copy()/np.sum(H)
plt.title("Linear span: "+str(x1)+" - "+str(x2))
plt.bar(sr, HP, 100/N)

plt.subplot(3,4,6)
plt.title("Function trans.")
fx = np.linspace(x1,x2,100)
plt.plot([0, x1, x2, 255], [0, 0, 255, 255])

plt.subplot(3, 4, 10)
plt.title("Image after: ")
plt.imshow(spannedlinear, cmap='gray', vmin=0, vmax=255)
# # #

#NON LINEAR SPAN: hist & img # # #
plt.subplot(3,4,3)
spannednonlinear,fy = non_linear_span(image, x1, x2, "power")
H,sr = histogram(spannednonlinear, N)
HP = H.copy()/np.sum(H)
plt.title("NON-lin span: "+str(x1)+" - "+str(x2))
plt.bar(sr, HP, 100/N)

plt.subplot(3,4,7)
plt.title("Function trans.")
fx = np.linspace(x1,x2,100)
bx = np.linspace(0, x1, 100)
by = np.ones(bx.shape[0])*np.amin(fy)
plt.plot(bx, by, "C0")
px = np.linspace(x2, 255, 100)
py = np.ones(px.shape[0])*np.amax(fy)
plt.plot(px, py)
plt.plot(fx, fy, "C0")

plt.subplot(3, 4, 11)
plt.title("Image after: ")
plt.imshow(spannednonlinear%255, cmap='gray', vmin=0, vmax=255)
# # #

#HISTOGRAM EQUALIZATION: hist & img # # # 
plt.subplot(3,4,4)
H,sr = histogram(image, N)
H1,equalized = equalization_hist(image, H)
HP = H1.copy()/np.sum(H1)
plt.title("hist equalization")
plt.bar(sr, HP, 100/N)

plt.subplot(3, 4, 12)
plt.title("Image after: ")
plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
# # #

plt.subplots_adjust(left=0.025, wspace=0.3)
plt.show()