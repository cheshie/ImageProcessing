import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
#######################################################

def okrag(R):

    theta_range = np.arange(0,2*np.pi,0.01*np.pi);
    
    M = np.zeros((2*R+1, 2*R+1));
    
    for theta in theta_range:
        
        x = R + R * np.cos(theta);
        y = R + R * np.sin(theta);
        
        x = np.round(x).astype('int');
        y = np.round(y).astype('int');

        M[x,y] = 1; 
            
    return M;


## Hougha ##
def ht(oimg, rmin, rmax, N):

    wybrany_prog = 15; 
    radius_step = 1;
    
    prog_img = np.sqrt((np.power(ffilter(oimg,maskL),2) + np.power(ffilter(oimg,maskL1),2)));

    # # # Progowanie obrazu
    po_img = (prog_img >= wybrany_prog).astype('float64');

    # Akumulator tworzenie
    AKU = np.zeros((po_img.shape[0],po_img.shape[1]));
    WSP_OKREGOW = np.zeros((po_img.shape[0],po_img.shape[1]));
    FINTAB = np.zeros((po_img.shape[0],po_img.shape[1]));
    AKUSUM = AKU;
        
        #Wypelnianie akumulatora zgodnie z wsp. krawedzi w obrazie
    radius_range = np.arange(rmin, rmax, radius_step);
        #rows, cols = np.where(po_img == 1);
        
    for R in radius_range:
            #Narysuj okrag w masce i przefiltruj nim obraz
        MA = okrag(R);
        AKU = ffilter(po_img, MA);
            
            #warindex = 0;
            #WARX = [0] * (AKU.shape[0]);
            #COORDX = [0] * (AKU.shape[0]);
            
            #Zalezenie od ilosci szukanych okregow
        for poz in range(N):
                
                #Dla AKU znajdz maksymalna wartosc
            w,k = np.where(AKU == np.amax(AKU))
                
                #Wpisz maksymalne wartosci i ich wsp do nowej tabeli
            if WSP_OKREGOW[w,k].any() < AKU[w,k].any():
                    #Prawdopodobienstwo (to dzielenie)
                WSP_OKREGOW[w,k] = AKU[w,k] / (2 * np.pi * R);
                #Usun znaleziony wynik
                AKU[w,k] = 0.0;
                 
        #WSP_OKREGOW zawiera duzo wiecej wsp niz N. Nalezy je odfiltrowac
        #I wybrac tylko te 10 najbardziej prawdopodobnych
        for poz in range(N):
            w,k = np.where(WSP_OKREGOW == np.amax(WSP_OKREGOW))
                
            if FINTAB[w,k].any() < WSP_OKREGOW[w,k].any():
                FINTAB[w,k] = 1.0;
            
            WSP_OKREGOW[w,k] = 0.0;
            
    plt.figure("Circle detection", figsize=(9,5))
    plt.subplot(1,2,1)
    plt.title("Original image")
    plt.imshow(po_img, cmap='gray');
        
    plt.subplot(1,2,2)
    plt.title("Detected")
    plt.imshow(FINTAB, cmap='gray');
    plt.show(block='false');
    
    return 1
#

image_circles = np.asarray(Image.open('Hough_okregi.png'))[:,:,0]
ht(image_circles, 50, 60, 2)

# FOR 10, for 2 N as well
# TESTY: 20 - 30 OKAY
# TESTY: 5  - 30 NOT OKAY
# TESTY: 50 - 60 OKAY


    