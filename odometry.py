import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os

class VisualOdometry() :
    """
    Visual odometry class to extract movement from two rectivied images.
    Error depends on configurations, e.g.: 0.308% (tested on 36,000 examples)
    [Python 3.6 and OpenCv2 4.4]
    """

    def __init__(self) :

        self.threshold = 0.4        #threshold for confidence score (when less then 40% right, then recursive call with more iterations)
        self.maxItertations = 50    #number of maximal iterations for recursive execution

    def getMovement(self, frame1, frame2, its=4) :
        """
        Recursive Monte-Carlo template matching. Create random patches in image and perfrom template matching. When
        confidence score lower then a threshold, recursivley call method with increased number of patches, until threshold or maximum
        number of iterations are achieved. Returns the predicted x-y translation, a confidence score, and the number of iterations for
        template matching with different patches.
        """

        img = frame1/255

        diffs = []
        for i in range(its) :

            template, x1, y1 = self.getPatch(frame2)            #create random size and random location patch

            res = cv2.matchTemplate(img,template,4)             #use of method TM_CCOEFF based on https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695daf9c3ab9296f597ea71f056399a5831da
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            diff = np.array(max_loc) - np.array([y1, x1])       #highest probability max_loc and offset result in predicted translation
            diffs.append(diff)

        diffs = np.array(diffs)
        element, count = np.unique(diffs, axis = 0, return_counts=True) #extract unique tuples and their count

        percent = count[np.argmax(count)]/its
        result = element[np.argmax(count)]

        if percent<= self.threshold :       #threshold of most identical predictions
            its += 4
            if its <= self.maxItertations :
                result, percent, its = self.getMovement(frame1, frame2, its)    #recursive call with higher number of iterations

        return result, percent, its

    def getPatch(self, frame) :
        """
        Extract random patch for block matching
        """

        w = frame.shape[0]
        h = frame.shape[1]

        x1 = np.random.randint(0,w-5)           #ensure that patch is of non-zero size
        x2 = np.random.randint(x1+1,w)          #ensure that x2>x1

        y1 = np.random.randint(0,h-5)
        y2 = np.random.randint(y1+1,h)

        template = frame[x1:x2, y1:y2]/255

        return template, x1, y1     #return of patch and position

    def createBatch(self) :
        """
        Create a batch of rectified image pairs based on random images of a folder, and random image offset
        Intended for testing
        """

        imgs = []
        labels = []

        imgPath = './imgs'
        imgList = os.listdir(imgPath)

        ran = np.random.choice(imgList)
        path = os.path.join(imgPath, ran)
        img = Image.open(path).convert('1')

        offset = np.random.randint(0,50,size=(2))
        pos = np.random.randint(0,img.size[1],size=(2))
        dim = 200 # is double of crop size

        #left, upper, right, lower  img.rotate(25).crop((100, 300, 200, 400))
        img1 = img.crop((int(pos[0]-dim), int(pos[1]-dim), int(pos[0] + dim), int(pos[1]+dim)))
        img2 = img.crop((int(pos[0]+offset[0]-dim), int(pos[1]+offset[1]-dim), int(pos[0]+offset[0] + dim), int(pos[1]+offset[1]+dim)))

        img1 = np.array(img1, dtype = np.float32)
        img2 = np.array(img2, dtype = np.float32)

        return img1, img2, offset


    def test(self) :
        """
        Simple test for movement function
        """
        num = 100
        for i in range(num) :

            img1, img2, offset = self.createBatch()                 #create pairs of rectified images
            result, percent, its = self.getMovement(img1, img2)     #determine translation

            err = np.mean(np.abs(result - offset))

            print(f'{percent}  {its}   {result}   {offset}  {err}')

#Initiate class and call test function
vis = VisualOdometry()
vis.test()



'''
Comments:

- Previously experimented with Optical flow of features, which is fast, but less accurate then the template matching method.
- I also experimented with a simple deep learning solution of convolutional and recurrent network, but that would be more
    relevant for real-live examples with illumination variation, occlusion variation, pose, etc. (Out of the scope of this assignment)
- For a fast configuration the error is 3.08% for 36,000 tests, can be improved by changing configs, but will impair performance.
- Transforms to black-white images for template matching. (Can be applied channel-wise for certain cases, e.g. low confidence)
- Monte-Carlo approach of different size patches increases the chances of correct prediction and
- Assumption of no need for optimization for parralel computation. (Could use threads)
- Assumption that no error handling necessary. (E.g. Method just called after validation of correct images)

'''
