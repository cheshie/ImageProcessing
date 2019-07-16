import numpy as np
import matplotlib.pyplot as plot
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



def dodawanie(obrt, nr):
    obrt=(obrt+nr) % 255

    return obrt


def mnozenie(obrt,nr):
    obrt=(obrt*nr) % 255

    return obrt


def potegowanie(obrt,nr):
    obrt=np.power(obrt, nr) % 255

    return obrt


def pierwiastek(obrt):
    obrt=np.sqrt(obrt) % 255

    return obrt


def logarytm(obrt):
    obrt = np.log10(obrt) % 255

    return obrt


#  MAIN:

image = np.asarray(Image.open('litery_1.png'))[:,:,0]
j,k = jasnosc_kontrast(image)

plot.subplot(2, 3, 1)
org_str = "original: "+str(j)+' : '+str(k)
plot.title(org_str)
plot.imshow(image, cmap='gray', vmin=0, vmax=255)

plot.subplot(2, 3, 2)
dod=dodawanie(image, 100)
j,k = jasnosc_kontrast(dod)
dod_str = "addition: "+str(j)+' : '+str(k)
plot.title(dod_str)
plot.imshow(dod % 256, cmap='gray', vmin=0, vmax=255)

plot.subplot(2, 3, 3)
mn=mnozenie(image, 10)
j,k = jasnosc_kontrast(mn)
mn_str = "multiply: "+str(j)+' : '+str(k)
plot.title(mn_str)
plot.imshow(mn % 256, cmap='gray', vmin=0, vmax=255)

plot.subplot(2, 3, 4)
p=potegowanie(image, 100)
j,k = jasnosc_kontrast(p)
pot_str = "power: "+str(j)+' : '+str(k)
plot.title(pot_str)
plot.imshow(p % 256, cmap='gray', vmin=0, vmax=255)

plot.subplot(2, 3, 5)
pier = pierwiastek(image)
j,k = jasnosc_kontrast(pier)
pie_str = "sqrt: "+str(j)+' : '+str(k)
plot.title(pie_str)
plot.imshow(pier % 256, cmap='gray', vmin=0, vmax=255)

plot.subplot(2, 3, 6)
log = logarytm(image)
j,k = jasnosc_kontrast(log)
log_str = "log: "+str(j)+' : '+str(k)
plot.title(log_str)
plot.imshow(log % 256, cmap='gray', vmin=0, vmax=255)

plot.show()