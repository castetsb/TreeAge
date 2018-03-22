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

import skimage.filters as filters

class TreeAge():
    """
    Class used to calculate the age of a tree from an image of its slice and
    the coordinate of the slice center on the image.
    """
    def __init__(self,img,detailSteps=False):
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
        self.center=None
        
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
        #Two methods are available to find slice center
        #V1 is fast and accurate but looks easy to fail with noise
        #V2 is low and less accurate but looks more robust to noise
        self._findCenterV1()
        #self._findCenterV2()
        self._findRadius()
        self._radialStack()
        self._bestProfile()
        self._dftAnalysis()
    
    def _insideSquare(self):
        """Return center and size of the inside fit circle of the tree slice
        
        Output:
        *******
            center:
                Center of the biggest inside fit square (n,p)
            size:
                Size of the biggest inside fit square
        """
        #Find the most distant point to tree slice border using distance map
        distMap=ndimage.distance_transform_edt(self.imgMask)
        center=ndimage.maximum_position(distMap)    
        #radius of the biggest inside fit circle
        radius=distMap[center]
        #Calculation of the size of the biggest inside fit square
        size=np.sqrt(2)*radius
        
        return center,size
    
    def _findCenterV1(self):
        """Find center of the tree slice in the input image using gradient.
        """
        #1) Find the biggest square fitting inside the slice
        fitSquareCenter,fitSquareSize=self._insideSquare()
        
        #2) Define square where layers orientation is analyzed to find slice
        #center
        
        #The square where is performed analysis is 2 time smaller than the
        #biggest inside fit square
        analysisSquareSize=int(fitSquareSize//2)
        halfSize=int(analysisSquareSize//2)
        analysisSlice=(np.s_[fitSquareCenter[0]-halfSize:fitSquareCenter[0]+halfSize],
                       np.s_[fitSquareCenter[1]-halfSize:fitSquareCenter[1]+halfSize])
        
        #3) Find tree slice center in the analysis square
        center=self._sliceCenterV1(analysisSlice)
        
        #4) Save center
        self.center=center
        """
        #Second try with smaller square
        #Failed probably because noise is dominant
        halfSize=int(halfSize//2)
        analysisSlice=(np.s_[center[0]-halfSize:center[0]+halfSize],
                       np.s_[center[1]-halfSize:center[1]+halfSize])
        center=self._sliceCenterV1(analysisSlice)
        """
        
    def _sliceCenterV1(self,analysisSlice):
        """Find center of the tree slice in the input slice using gradient.
        
        Parameters
        **********
        analysisSlice:
            np.s_ slice defining the region where to search for tree slice
            center.
        
        Output
        ******
        center:
            center of the tree slice (n,p)
        """
        #Origin of the square of analysis
        n0=analysisSlice[0].start
        p0=analysisSlice[1].start
        
        #Image in the square
        img=self.img[analysisSlice]
        
        #Vertical gradient
        vGrad=ndimage.sobel(img.astype(np.int),axis=0)
        
        #Horizontal gradient
        hGrad=ndimage.sobel(img.astype(np.int),axis=1)
        
        #Average gradient per line and column
        averageVGradByLine=np.average(np.abs(vGrad),1)
        averageHGradByColumn=np.average(np.abs(hGrad),0)
        
        #Filtering of harmonic 1 to extract global minimum which should be
        #center position
        dftN=np.fft.fft(averageVGradByLine)
        dftN[1:-1]=0
        profN=np.real(np.fft.ifft(dftN))
        centerN=int(np.argmin(profN))
        
        dftP=np.fft.fft(averageHGradByColumn)
        dftP[1:-1]=0
        profP=np.real(np.fft.ifft(dftP))
        centerP=int(np.argmin(profP))
        
        #Build center
        center=(n0+centerN,p0+centerP)
        
        if self.detailSteps:
            print("Estimated center position: {}".format(center))
            print("Real center position: {}".format((1959,1928)))
            print("Error : ({},{})".format(np.abs(1959-center[0]),np.abs(1928-center[1])))
            #plotting
            plt.subplot(2,2,1)
            plt.title("Average vertical gradient by line")
            y=np.linspace(0,averageVGradByLine.size-1,averageVGradByLine.size)
            plt.gca().invert_yaxis()
            plt.plot(averageVGradByLine,y)
            plt.plot(profN,y)
            plt.subplot(2,2,2)
            plt.title("Square used for analysis")
            plt.imshow(img)        
            plt.subplot(2,2,4)
            plt.title("Average horizontal gradient by column")
            plt.plot(averageHGradByColumn)
            plt.plot(profP)
            plt.show()
        
        return center
    def _findCenterV2(self):
        """
        Find center of the tree slice in the input image by thresholding layers.
        """
        #1) Find the biggest square fitting inside the slice
        fitSquareCenter,fitSquareSize=self._insideSquare()
        
        #2) Define square where layers orientation is analyzed to find slice center
        
        #The square where analysis is perform is 2 time smaller than the
        #biggest inside fit square
        #The side of the square is adjusted to be perfectly divisible in 9
        #sub-square with each sub-square having an odd number of pixels to have
        #one pixel at their center.
        analysisSquareSize=int(((fitSquareSize//2)//9)*9)
        #Build square slice
        analysisSlice=(np.s_[fitSquareCenter[0]-analysisSquareSize//2:fitSquareCenter[0]+analysisSquareSize//2],
                       np.s_[fitSquareCenter[1]-analysisSquareSize//2:fitSquareCenter[1]+analysisSquareSize//2])
        
        #3) First analyse center position in the analysis square
        center=self._sliceCenterV2(analysisSlice)
        
        #4) Renew analysis to rafine center position
        
        #Record of center position for each analysis
        if self.detailSteps:
            centersN=[]
            centersP=[]
            centersN.append(center[0])
            centersP.append(center[1])
        
        #Renew analysis until convergence or reach 20 loops
        i=0
        converge=False
        while i<20 and not converge:
            #Test converge to a point
            #We consider that we converged to a point if the estimated center
            #position moved for less than 5 px on the last 4 analysis.
            if i>3:
                centerNDelta=np.max(centersN[-4:])-np.min(centersN[-4:])
                centerPDelta=np.max(centersP[-4:])-np.min(centersP[-4:])
                if centerNDelta<5 and centerPDelta<5:
                    converge=True
            #Redefinition of the square of analysis
            analysisSlice=(np.s_[center[0]-analysisSquareSize//2:center[0]+analysisSquareSize//2],
                           np.s_[center[1]-analysisSquareSize//2:center[1]+analysisSquareSize//2])
            #Analyse center position in the analysis square
            center=self._sliceCenterV2(analysisSlice)
            #Save result of the analysis
            centersN.append(center[0])
            centersP.append(center[1])
            i+=1
        
        #Save center
        self.center=center
        
        if self.detailSteps:
            print("Estimated center position: {}".format(center))
            print("Real center position: {}".format((1959,1928)))
            print("Error : ({},{})".format(np.abs(1959-center[0]),np.abs(1928-center[1])))
            centersN=np.array(centersN)
            centersP=np.array(centersP)
            plt.subplot(2,2,1)
            plt.title("Displacement step of center n /n coordianate over iteration")
            plt.plot(centersN[:-1]-centersN[1:])
            plt.subplot(2,2,2)
            plt.title("Displacement step of center p /n coordianate over iteration")
            plt.plot(centersP[:-1]-centersP[1:])
            plt.subplot(2,2,3)
            plt.title("Total displacement of center n /n coordianate over iteration")
            plt.plot(centersN[0]-centersN[1:])
            plt.subplot(2,2,4)
            plt.title("Total displacement of center p /n coordianate over iteration")
            plt.plot(centersP[0]-centersP[1:])
            plt.show()
    
    def _sliceCenterV2(self,analysisSlice):
        """
        Find center of the slice in the input slice by thresholding layers.
        
        Parameters:
        ***********
        analysisSlice:
            np.s_ slice defining the region where to search for tree slice
            center.
        
        Output
        ******
        center:
            center of the tree slice (n,p)
        """
        #Origin of the analysis square
        n0=analysisSlice[0].start
        p0=analysisSlice[1].start
        
        
        #Image in the analysis square
        analysisImg=self.img[analysisSlice]
        
        #1) Define sub square
        
        #The analysis square is divided in 9 sub-square
        #Size of a sub-square
        subSquareSize=analysisImg.shape[0]//3        
        """ 
        #Plotting put in comments because the functions is called many times
        if self.detailSteps:
            plt.title("Image used to find center position")
            plt.imshow(analysisImg)
            plt.show()
        """
        
        #Sub square images
        subSquareImgs=[]
        #Sub square centers in analysis square referential
        subSquareCenters=[]
        
        #Get sub square picture and center
        i=0
        while i<3:
            #n range of the sub square
            nRange=np.s_[0+i*subSquareSize:subSquareSize+i*subSquareSize]
            j=0
            while j<3:
                #p range of the sub square
                pRange=np.s_[0+j*subSquareSize:subSquareSize+j*subSquareSize]
                #Image of the sub square
                subSquareImg=analysisImg[nRange,pRange]
                #Sub square center
                center=(subSquareSize//2+i*subSquareSize,subSquareSize//2+j*subSquareSize)
                #Save sub square image
                subSquareImgs.append(subSquareImg)
                #Save sub square center
                subSquareCenters.append(center)
                """
                #Plotting put in comments because the functions is called many times
                if self.detailSteps:
                    plt.subplot(3,3,i*3+j+1)
                    plt.title("Path {}".format(i*3+j+1))
                    plt.imshow(path)
                """
                j+=1
            i+=1
        """
        #Plotting put in comments because the functions is called many times
        if self.detailSteps:
            plt.show()
        """
        #2) Find lines perpendicular to sub square layers for each sub square
        
        #List where are saved lines perpendicular to sub square layers
        lines=[]
        #Analysis each sub square
        i=0
        while i<len(subSquareImgs):
            #Sub square image
            subSquareImg=subSquareImgs[i]
            #Sub square center
            center=subSquareCenters[i]
            #Get the angle of the line perpendicular to sub square layers
            angle=self._layerAngle(subSquareImg)
            #Equation of the line perpendicular to sub square layers
            #Line equation: n=-ap+b
            #Director coefficient: a
            a=np.tan(angle)
            #Intersept: b
            b=int(center[0]+a*center[1])
            #Save parameters of line perpendicular to sub square layers
            lines.append((a,b))
            i+=1
        
        #3) Accumulate sub squares lines inverse distance map in an accumulator
        #image.
        
        #p range used to draw line
        p=np.arange(0,analysisImg.shape[1],1)
        #Distance map accumulator image
        accumulator=np.zeros(analysisImg.shape)
        #Maximum possible distance used to build distance map inverse
        accMax=int(np.sqrt(analysisImg.shape[0]**2+analysisImg.shape[1]**2)+1)
        
        #Distance map accumulation loop
        i=0
        for line in lines:
            #Calculation of line points n coordinates
            n=(-1*line[0]*p+line[1]).astype(np.int)
            #Remove points out of the analysis image (n<0 or n>analysisImg.shape[0])
            nFilter=(n>=0)*(n<analysisImg.shape[0])
            #Filtered coordinates
            n2=n[nFilter]
            p2=p[nFilter]
            #Points of the line
            pts=pts=(n2,p2)
            #Build line mask
            mask=np.ones(analysisImg.shape,np.bool)
            mask[pts]=False
            #Build line inverse distance map
            distMap=accMax-ndimage.distance_transform_edt(mask)
            """
            #Plotting put in comments because the functions is called many times
            if self.detailSteps:
                plt.imshow(distMap)
                plt.show()
            """
            #Add line distance map to the accumulator
            accumulator=accumulator+distMap
            i+=1
        #Position of the tree slice center in the analysis square
        localCenter=ndimage.maximum_position(accumulator)
        #Global center position
        center=(n0+localCenter[0],p0+localCenter[1])
        """
        #Plotting put in comments because the functions is called many times
        if self.detailSteps:
            print("Max :",np.max(accumulator))
            print("Calculated center: {}".format(center))
            plt.imshow(accumulator)
            plt.show()
        """
        return center
        
    
    def _layerAngle(self,img):
        """
        Return the average angle of the direction perpendicular to iamg
        layers.
        
        Parameters:
        ***********
        img:
            ndimage grey scale image
            
        Output:
        *******
        angle:
            Angle of angle of the direction perpendicular to layers OR None if
            the detection failed.
            Angle is in range [0,Pi]
        """
        #1) Otsu threshold of the input image to highlight layers
        imgOtsu=img<filters.threshold_otsu(img)
        
        #Label thresholded regions
        labels,nbrLabels=ndimage.label(imgOtsu)
        """
        #Plotting put in comments because the functions is called many times
        if self.detailSteps:
            plt.imshow(labels)
            plt.show()
        """
        
        #1) Filter regions with bounding big diagonal bigger than 40% of sub square size
        
        #Find box of each labeled regions
        labelsBoxs=ndimage.find_objects(labels)
        
        #Array used to filter labels
        #A value of 0 means that the corresponding label is removed
        bigLabels=np.zeros(nbrLabels+1)
        i=0
        while i<nbrLabels:
            #Box where is located the region to be filtered
            box=labelsBoxs[i]
            #Region box size
            nSize=box[0].stop-box[0].start
            pSize=box[1].stop-box[1].start
            #Region box diagonal
            diag=np.sqrt(nSize**2+pSize**2)
            #Check diagonal size
            if diag>img.shape[0]*0.4:
                #Add the region if diagonal is more than 40% of sub square size
                bigLabels[i+1]=i+1
            i+=1
        #Apply label filter
        labels=bigLabels[labels]
        #Relabel
        labels,nbrLabels=ndimage.label(labels)
        """
        #Plotting put in comments because the functions is called many times
        if self.detailSteps:
            plt.imshow(labels)
            plt.show()
        """
        
        
        #2) Find regretion line for each labelled regions
        
        #In case there is no label, the function return None
        if nbrLabels==0:
            print("ERROR: No regions big enough to estimate layers perpendicular")
            return None
        #Array where are save labeled regions perpendicular direction angles
        angles=np.zeros(nbrLabels)
        #Loop over labelled regions
        i=0
        while i<nbrLabels:
            #Find labeled region points coordinates
            n,p=np.where(labels==i+1)
            #Linear regression of the labeled region
            line=np.polyfit(n,p,1)
            #Note: Angle of np.arctan function is in range [-pi/2, pi/2]
            #Calculate angle of direction perpendicular to regression line
            angle=np.arctan(line[0])
            #Adjust angle range to avoid having -pi/2 and pi/2 which represent
            #Same direction
            if angle==-np.pi/2:
                #Change angle intervale ]-pi/2, pi/2]
                angle=-angle
            #Save perpendicular angle
            angles[i]=angle
            i+=1
        #Calculate the average angle
        angle=np.angle(np.sum(np.cos(angles)+1j*np.sin(angles)))
        #Adjust angle range to [0,Pi[
        if angle<0:
            angle=np.pi+angle
        elif angle==np.pi:
            angle=0
        """
        #Plotting put in comments because the functions is called many times
        if self.detailSteps:
            print("The angle of the direction perpendicaular to layers is: {} degrees".format(np.degrees(angle)))
        """
        return angle
    
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
#Center position is (1959,1928)
app=TreeAge(imgGrey,True)
#Print result
print("The age of the tree is : {}".format(app.age))









        
        
    

