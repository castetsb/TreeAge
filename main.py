# -*- coding: utf-8 -*-
"""
Created by Benoit CASTETS
9/3/2018
Simple program to find the age of a tree from a picture of its slice
"""
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.stats as stats
from scipy.signal import butter, lfilter, freqz


###################################################################
#Pass filter
#https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
###################################################################

#Set DEMO to True to plot each steps of the analysis
DEMO=True

class TreeAge():
    """
    Class used to calculate the age of a tree from a picture of its slice and
    the coordinate of the slice center on the image.
    """
    def __init__(self,img,center):
        """
        Create a TreeAge application
        
        Parameters:
        ***********
        img:
            Grey scale img of the tree slice.
            The tree slice must be surrounded by white.
        center:
            tuple with tree slice center coordinate. (n,p)            
        """
        #Attributes
        ###########
        ###########
        
        #Imput
        ######
        self.img=img.copy()
        self.center=center
        
        #Intermediate
        #############
        #Binary mask for the slice
        self.imgMask=self.img<240
        #Tree boundary "box"
        self.treeSlice=ndimage.find_objects(self.imgMask)[0]      
        #Trimmed image (used for display)
        self.treeImg=img[self.treeSlice]
        #Radius of the slice
        self.radius=None      
        #Unrolled slice image
        self.sliceStack=None
        #Most representive profile
        self.bestProfil=None
        
        #Output
        #######
        self.age=None
        
        #Analysis launch
        ################
        ################
        self.analyse()
    
    
    def analyse(self):
        self._findRadius()
        self._unroll()
        self._bestProfile()
        #Maybe added in the future
        #self._filtering()
        self._findAge()
        
    def _findRadius(self):
        """
        Find the maximum distance between slice center and slice border point.
        This distance is considered as the slice radius.
        """
        #Find points on the border of the slice using
        kernel=np.ones((3,3),np.bool)
        erosionMask=ndimage.binary_erosion(self.imgMask,kernel)
        borderMask=self.imgMask>erosionMask
        borderPts=np.where(borderMask)
        nbrPts=borderPts[0].size
        
        #ploting
        if DEMO==True:
            plt.subplot(1,3,1)
            plt.title("Slice mask")
            plt.imshow(self.imgMask,cmap="gray")
            plt.subplot(1,3,2)
            plt.title("Erosion mask")
            plt.imshow(erosionMask,cmap="gray")
            plt.subplot(1,3,3)
            plt.title("Border mask")
            plt.imshow(borderMask,cmap="gray")
            plt.show()
        
        #Find the maximum distance between border points and center
        i=0
        radius=0
        while i<nbrPts:
            n=borderPts[0][i]
            p=borderPts[1][i]
            dist=np.hypot(n-self.center[0],p-self.center[1])
            if dist>radius:
                radius=dist            
            i+=1
        #unit troncature of the radius
        radius=int(radius)
        self.radius=radius
        
        #Print result
        if DEMO==True:
            print("The slice radius length is: {}px".format(self.radius))
    
    def _unroll(self):
        """
        Make a vertical stack image of radial profil of the tree slice.
        The profile is done every degree.
        
        self.radius must be known to use this function. Excute _findBdCercle
        as a first step.
        """
        #Make a loop to get profil on each degree
        i=0
        n0,p0=self.center
        nbrProfil=1700
        self.treeStack=np.zeros((nbrProfil,self.radius),np.uint8)
        while i<nbrProfil:
            deg=(360/nbrProfil)*i
            
            #profil extremity
            n1=int(n0+self.radius*math.sin(math.radians(deg)))
            p1=int(p0+self.radius*math.cos(math.radians(deg)))
            #Get pixel on the radius
            n, p = np.linspace(n0, n1, self.radius),np.linspace(p0, p1, self.radius)
            #Get value at each pixels
            prof=self.img[n.astype(np.int), p.astype(np.int)]
            
            
            #remove white pixels extend profil to maximum radius
            if np.sum(prof>240)>0:
                profEnd=np.argmax(prof>240)
                
                prof=prof[0:profEnd]
                idx=np.linspace(0,profEnd-1,self.radius)
                f=interpolate.interp1d(np.arange(0,prof.size,1),prof,kind="quadratic")
                profInterp=f(idx)
                self.treeStack[i,:] = profInterp
            else:
                self.treeStack[i,:] =prof
            i+=1
        if DEMO==True:
            plt.subplot(1,2,1)
            plt.title("Tree slice")
            plt.imshow(self.treeImg,cmap="gray")
            plt.subplot(1,2,2)
            plt.title("Stacked radial profiles")
            plt.imshow(self.treeStack,cmap="gray")
            plt.show()

    def _bestProfile(self):
        """
        Find the most representative profil in treeStack. This will be use
        for age calculation
        self.treeStack must be calculated before using this function.
        """
        #Find the profile with smallest anthopy in treeStack
        
        minEntropy=stats.entropy(self.treeStack[0,:])
        minEntropyN=0
        
        i=1
        while i<self.treeStack.shape[0]:
            entropy=stats.entropy(self.treeStack[i,:])
            if entropy<minEntropy:
                minEntropyN=i
                minEntropy=entropy
            i+=1
        
        self.bestProfile=self.treeStack[minEntropyN,:]
        if DEMO==True:
            print("The profile with minimum entropy is the at line {}/{}".format(minEntropyN,self.treeStack.shape[0]))
            plt.title("Best radial profile")
            plt.plot(self.bestProfile)
            plt.show()
        
    def _findAge(self):
        """
        """
        #Frequency band pass filtering
        #prof=butter_bandpass_filter(self.mainProfil,10,100,self.mainProfil.size)
        
        #find local minima
        localMinFilter=ndimage.filters.minimum_filter1d(self.bestProfile,30)
        localMaxFilter=ndimage.filters.maximum_filter1d(self.bestProfile,30)
        
        pickZone=(localMaxFilter-localMinFilter)>20
        
        localMinMask=(self.bestProfile==localMinFilter)*pickZone
        self.age=sum(localMinMask)
        
        if DEMO==True:
            print("The tree is {} years old.".format(self.age))
            plt.subplot(2,1,1)
            plt.title("Best radial profile")
            plt.plot(self.bestProfile)
            plt.subplot(2,1,2)
            plt.title("Local luminosity minima")
            plt.plot(localMinMask)
            plt.show()
    

#Test
imgColor=plt.imread("tree.JPG")
imgGrey=(np.sum(imgColor,2)/3).astype(np.uint8)



#cv2.imread("tree2.JPG",cv2.IMREAD_GRAYSCALE)
app=TreeAge(imgGrey,(1959,1928))
#app.showCenterCroosProfil()









        
        
    

