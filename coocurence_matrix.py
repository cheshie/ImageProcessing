# Coocurence matrix! 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def fCoocurence(img_c, N, dir):
    #Two dimensions: rows, cols. FROM:WHERE:STEP
    #If direction is negative, change order of array elems: 
    
    #rows: 
    if dir[0] < 0: 
        img_c = img_c[:,-1::-1]
        dir[0] = -dir[0]
    #cols:
    if dir[1] < 0: 
        img_c = img_c[-1::-1,:]
        dir[1] = -dir[1]
    # ^ changing order ^     
        
    
    #Creating matrices with coordinates - X
    if dir[0] != 0:
        Z = img_c[:,:-1:dir[0]]
        DO = img_c[:,1::dir[0]]         
    
    #Creating matrices with coordinates - Y    
    if dir[1] != 0:
        Z = img_c[:-1:dir[1],:]
        DO = img_c[1::dir[1],:]
        
    #Create matrix with the size of coordinates range
    cMatrix = np.zeros(( np.amax(Z)+1,np.amax(DO)+1 ))
#     cMatrix = np.zeros((N,N))
#     img_c = img_c / (256 - N+1)
    
    
    #Calculate elements of matrix
    #     cMatrix[Z.reshape(-1),DO.reshape(-1)] += 1
    np.add.at(cMatrix, [ [Z.flatten('F')],[DO.flatten('F')]], 1)
    return cMatrix
#

def blur_percentage(img):
    return np.sum(img > np.amin(img)) * (np.amax(img)) /  (img.shape[0]*img.shape[1])
#

# # # # MAIN: # # # # 
    
    
# sharp_image = plt.imread('Oko_sharp.png')
sharp_gray = np.asarray(Image.open('Oko_sharp.png'))
blurred_gray = np.asarray(Image.open('Oko_blurred.png'))
print()
print(np.amax(blurred_gray)*np.sum(blurred_gray == np.amax(blurred_gray)))
space_number = 10;
# sharp_gray = 0.299 * sharp_image[:,:,0] + 0.587 * sharp_image[:,:,1] + 0.114 * sharp_image[:,:,2] 

plt.figure("Coocurence Matrix")
plt.subplot(2,2,1)
sharp_string = "Sharp: "+str(round(blur_percentage(fCoocurence((sharp_gray), space_number, [1,0])),2))+"% blurred"
plt.title(sharp_string)

plt.imshow(sharp_gray, cmap='gray')

plt.subplot(2,2,2)
plt.title("Sharp - cMatrix")
# plt.imshow(fCoocurence(int(sharp_gray*255), 1, [1,0]))
plt.imshow(fCoocurence((sharp_gray), space_number, [1,0]), cmap='gray')

plt.subplot(2,2,3)
blurred_string = "Blurred: "+str(round(blur_percentage(fCoocurence((blurred_gray), space_number, [1,0])),2))+"% blurred"
plt.title(blurred_string)
plt.imshow(blurred_gray, cmap='gray')

plt.subplot(2,2,4)
plt.title("Blurred - cMatrix")
# plt.imshow(fCoocurence(int(sharp_gray*255), 1, [1,0]))
plt.imshow(fCoocurence((blurred_gray), space_number, [2,4]), cmap='gray')


plt.draw() # warn: plt.show() doesn't like waitforbuttonpress

plt.waitforbuttonpress()
plt.close()
    
# TESTING
# A = np.matrix('0,1,2,3 ; 4,5,6,7')
# A = np.array([ [0,1,2],[0,1,2] ])
# IN CASE: (1,0) <- tuple does not support item assigment (READ-ONLY)
# fCoocurence(A, 1, [1,0] )
# TESTING
