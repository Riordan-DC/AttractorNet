#! /usr/bin/env python3

from attractor import ATTRACTOR
import numpy as np
import cv2
import sys

"""

    RESILIANT PLACE RECOGNITION WITH ATTRACTOR NETWORK SEARCH
    
    I hope to develop a method for modeling a place recognition
    search that is resliant to incorrect matches of places.
    The model is based on the hebbian theory of neural activity.
    This theory is implimented in the 1-D attractor network implimentation
    I have developed for python. 
"""

def loadFrames(directory, n):
    frames = [] 
    capture = cv2.VideoCapture(directory)
    count = 0
    while True:
        (grabbed, frame) = capture.read()
        if grabbed and count < n:
            #frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),                    (64, 32))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = cv2.resize(frame, (32, 16), interpolation=cv2.INTER_LANCZOS4)
            #frame = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
            count += 1
        else:
            break
            #cv2.imshow(directory, frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
    #print("Frames from video {} loaded! size: {} kb".format(directory,
    #    sys.getsizeof(frames)/1024))
    return frames

class recog(object):
    def __init__(self, refImages):
        self.NE = len(refImages)
        #self.net = ATTRACTOR(self.NE, 11.5, 2.6)
        self.net = ATTRACTOR(self.NE, float(sys.argv[1]), float(sys.argv[2]))
        self.memories = [self.normalize_image(image) for image in refImages]

    def step(self, image, duration):
        matches = self.compare(image)
        # If match is None. Settle activity. Step with zero input
        if matches == None:
            E, diff = self.net.step(np.zeros(self.NE))
            return E, diff
        E = self.net.E
        if np.argmax(E) == matches[0]:
            self.net.E = np.roll(E, 1)
            return self.net.E, None
        packet = np.zeros(self.NE)
        packet[matches] = 1
        E = diff = 0
        for activity in range(duration):
            E, diff = self.net.step(packet)
            """
            plt.clf()
            plt.ylim((0,1))
            plt.stem(E, markerfmt='C0.')
            plt.pause(0.05)
            """
        #E = np.clip(E, 0, 0.45)
        print("MATCH {} - PREDICTION {}".format(matches, np.argmax(E)))
        return E, diff

    def normalize_image(self, image):
        """
        image = np.sum(image, axis=0)
        image = np.true_divide(image, np.sum(image))
        """
        image = self.applyLocalContrast(image)
        return image

    def applyLocalContrast(self, frame):
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(4,4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(frame)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)/256
        return frame 

    def compare(self, image):
        """
        Todo:
        If the match isnt good enough, blow a threshold, dont return any match.
        Only return None.
        """
        threshold = 0.5
        queryImage = self.normalize_image(image)
        error = np.ones((self.NE)) 
        for index in range(len(self.memories)):
            error[index] = np.sum(np.absolute(queryImage - self.memories[index]))
        no_matches = 2
        matches = []
        for place in range(no_matches):
            best_match = np.argmin(error)
            print("Match strength {}".format(error[best_match]))
            if error[best_match] > threshold:
                matches.append(best_match)
                error[matches[-1]] = 2
        #print("Top {} matches are/is : {}".format(no_matches, matches))
        return matches 

    def settle_activity(self, E, threshold):
        E = np.zeros(self.NE)
        while True:
            difference = E
            E,_ = self.net.step(E)
            change = np.sum(np.absolute(difference - E))
            if change < threshold:
                return E


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import time
    refFrames = loadFrames('./day_trunc.avi', 200)
    queryFrames = loadFrames('./night_trunc.avi', 200)
    refFrames = refFrames[75:105] #-5]
    queryFrames = queryFrames[84:115]
    #queryFrames = refFrames
    
    RE = recog(refFrames)
    RE.settle_activity(RE.net.E, 0.001)
    results = np.zeros((len(refFrames), len(queryFrames)))
    matches = 0
    for img in range(len(queryFrames)):
        E,diff = RE.step(queryFrames[img], 4)
        #E = RE.settle_activity(E, 0.01)
        for i in range(15):
            E,_ = RE.net.step(E)
        results[:,img] = E
        prediction = np.argmax(E)
        plt.clf()
        plt.ylim((0,1))
        plt.stem(E, markerfmt='C0.')
        plt.pause(0.1)
        if prediction == img:
            matches += 1
            #print("%d SUCCSESS %d" % (img, prediction))
        else:
            pass
            #print("%d FAIL %d" % (img, prediction))
    plt.clf()
    plt.ylim((0,1))
    plt.stem(E, markerfmt='C0.')
    plt.show()
    print("Score = %d/%d (%f)" % (matches, len(queryFrames), (matches/len(queryFrames))))
