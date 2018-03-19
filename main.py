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

class TreeAge():
    """
    Class used to calculate the age of a tree from an image of its slice and
    the coordinate of the slice center on the image.
    """
    def __init__(self,img,center,detailSteps=False):
        """
        Create a TreeAge object.
        
        Parameters:
        ***********
        img:
            Grey scale ndimage of a tree slice.
        center:
            Tuple of slice center coordinate on the input image. (n,p) 
        detailsSteps:
            If True each steps of analysis are illustrated with plots.
        """
        #Attributes
        ###########
        ###########
        
        #options
        ########
        self.detailSteps=detailSteps
        
        #Input
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
        #Maximum radius of the slice
        self.radius=None      
        #Array where are stacked radial profiles
        self.sliceStack=None
        #Most representive profile
        self.bestProfile=None
        
        #Output
        #######
        self.age=None
        
        #Analysis launch
        ################
        ################
        self.analyse()
    
    def analyse(self):
        self._findRadius()
        self._radialStack()
        self._bestProfile()
        self._dftAnalysis()
        
    def _findRadius(self):
        """
        Find the maximum radius of the slice.
        """
        #Find points on the border of the slice using
        kernel=np.ones((3,3),np.bool)
        erosionMask=ndimage.binary_erosion(self.imgMask,kernel)
        borderMask=self.imgMask>erosionMask
        borderPts=np.where(borderMask)
        nbrPts=borderPts[0].size
        
        #ploting
        if self.detailSteps==True:
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
        if self.detailSteps==True:
            print("The slice radius length is: {}px".format(self.radius))
    
    def _radialStack(self):
        """
        Make a vertical stack image of radial profiles of the tree slice.
        """
        n0,p0=self.center
        #Number of profiles to be extracted
        nbrProfil=self.radius
        #Prepare array to save profiles
        self.treeStack=np.zeros((nbrProfil,self.radius),np.uint8)
        #Make a loop to attract radial profiles
        i=0
        while i<nbrProfil:
            #Angle of the radius
            deg=(360/nbrProfil)*i
            #Profile extremity
            n1=int(n0+self.radius*math.sin(math.radians(deg)))
            p1=int(p0+self.radius*math.cos(math.radians(deg)))
            #Pixel on the radius
            n, p = np.linspace(n0, n1, self.radius),np.linspace(p0, p1, self.radius)
            #Value of each pixels on the radius
            prof=self.img[n.astype(np.int), p.astype(np.int)]
            #If the profile ends out of the slice. We trim the profile portion out of the
            #slice and resize it to have same length as the maximum slice radius.
            if np.sum(prof>240)>0:
                #Trim the profile portion out of the slice.
                profEnd=np.argmax(prof>240)
                prof=prof[0:profEnd]
                #Resize the profile to have same length as the maximum slice radius.
                idx=np.linspace(0,profEnd-1,self.radius)
                f=interpolate.interp1d(np.arange(0,prof.size,1),prof,kind="quadratic")
                profInterp=f(idx)
                #Save extracted profile
                self.treeStack[i,:] = profInterp
            else:
                #Save extracted profile
                self.treeStack[i,:] =prof
            i+=1

        #Print results
        if self.detailSteps==True:
            plt.subplot(1,2,1)
            plt.title("Tree slice")
            plt.imshow(self.treeImg,cmap="gray")
            plt.subplot(1,2,2)
            plt.title("Stacked radial profiles")
            plt.imshow(self.treeStack,cmap="gray")
            plt.show()

    def _bestProfile(self):
        """
        Find the less noisy profil which will be use for analysis.
        We will consider it is the profil with the smallest entropy.
        """ 
        #Make a look to calculate entropy of each profiles
        #Find maximum (for ploting) and minimum entropy
        minEntropy=stats.entropy(self.treeStack[0,:])
        minEntropyN=0
        maxEntropy=stats.entropy(self.treeStack[0,:])
        maxEntropyN=0
        i=1
        while i<self.treeStack.shape[0]:
            entropy=stats.entropy(self.treeStack[i,:])
            if entropy<minEntropy:
                minEntropyN=i
                minEntropy=entropy
            if entropy>maxEntropy:
                maxEntropyN=i
                maxEntropy=entropy
            i+=1
        #Save the best profile
        self.bestProfile=self.treeStack[minEntropyN,:]
        
        if self.detailSteps==True:
            print("The profile with minimum entropy is the at the line {}/{}".format(minEntropyN,self.treeStack.shape[0]))
            print("The profile with maximum entropy is the at the line {}/{}".format(maxEntropyN,self.treeStack.shape[0]))
            plt.subplot(2,1,1)
            plt.title("Worst radial profile")
            plt.plot(self.treeStack[maxEntropyN,:])
            plt.subplot(2,1,2)
            plt.title("Best radial profile")
            plt.plot(self.bestProfile)
            plt.show()
    
    def _dftAnalysis(self):
        """
        Use Fourier DFT to find the domain harmonique which should be the age
        of the tree.
        """
        prof=self.bestProfile
        profDft=np.fft.fft(prof)
        profDftAbs=np.absolute(profDft)
        #We do not consider the 10 first harmonique
        #The 10 first harmonique are responsible for global luminosity variation
        #In simple word we are making the hypothesis that the tree is more than
        #10 year old.
        #With the DFT we can find age between 10 and maxRadius/2=879
        self.age=np.argmax(profDftAbs[10:prof.size//2])+9
        if self.detailSteps==True:
            x=np.arange(0,prof.size,1)
            print("The age of the tree is : {}".format(self.age))         
            plt.subplot(4,1,1)
            plt.title("Profile")
            plt.plot(prof)
            plt.subplot(4,1,2)
            plt.title("DFT absolute 0-9")
            plt.plot(profDftAbs[:10])
            plt.subplot(4,1,3)
            plt.title("DFT absolute 10-nyquist Frequency")
            plt.plot(x[10:prof.size//2],profDftAbs[10:prof.size//2])
            plt.subplot(4,1,4)
            plt.title("DFT absolute close up 10-100")
            plt.plot(x[10:101],profDftAbs[10:101])
            plt.show()

#Test       
imgColor=plt.imread("tree.JPG")
#Convert image to grey scale
imgGrey=(np.sum(imgColor,2)/3).astype(np.uint8)
#Create a TreeAge application
app=TreeAge(imgGrey,(1959,1928),True)
#Print result
print("The age of the tree is : {}".format(app.age))









        
        
    

