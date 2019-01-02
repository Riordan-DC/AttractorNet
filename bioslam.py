#! /usr/bin/env python3

from attractor import ATTRACTOR
import numpy as np
import cv2
import sys

def loadFrames(directory, n):
    frames = [] 
    capture = cv2.VideoCapture(directory)
    count = 0
    while True:
        (grabbed, frame) = capture.read()
        if grabbed and count < n:
            #frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),                    (64, 32))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (32, 16), interpolation=cv2.INTER_LANCZOS4)
            #frame = cv2.resize(frame, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)
            count += 1
        else:
            break
            #cv2.imshow(directory, frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
    print("Frames from video {} loaded! size: {} kb".format(directory,
        sys.getsizeof(frames)/1024))
    return frames

def build_packet(NE, location, active, length):
    packet = np.zeros((length, NE))
    packet[np.arange(active),location] = 1
    return packet

class recog(object):
    def __init__(self, refImages):
        self.NE = len(refImages)
        #self.net = ATTRACTOR(self.NE, 13.4, 3.28)
        self.net = ATTRACTOR(self.NE, 1, 1)
        self.memories = [self.process_image(image) for image in refImages]

    def step(self, image):
        matches = self.compare(image)
        packet = self.build_packet(matches, 1)
        E = diff = 0
        for activity in packet:
            E, diff = self.net.step(activity)
        #self.settle_activity(3, E)
        #E = np.clip(E, 0, 0.85)
        return E, diff

    def process_image(self, image):
        image = np.sum(image, axis=0)
        image = np.true_divide(image, np.sum(image))
        return image

    def compare(self, image):
        #Perform SAD across all images. 
        #Choose image with best score.
        queryImage = self.process_image(image)
        error = np.ones((self.NE)) 
        for index in range(len(self.memories)):
            error[index] = np.sum(np.absolute(queryImage - self.memories[index]))
        
        no_matches = 1
        matches = []
        for place in range(no_matches):
            matches.append(np.argmin(error))
            error[matches[-1]] = 2

        #print("Top {} matches are/is : {}".format(no_matches, matches))
        return matches 

    def build_packet(self, location, active):
        # Todo: Build an actual distribution of a activity
        packet = np.zeros((active, self.NE))
        packet[:,location] = 1
        return packet

    def settle_activity(self, iterations, initial_state):
        E = initial_state
        for i in range(iterations):
            E, diff = self.net.step(E)

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import time
    refFrames = loadFrames('./day_trunc.avi', 200)
    queryFrames = loadFrames('./night_trunc.avi', 200)
    refFrames = refFrames[75:85] #-5]
    queryFrames = queryFrames[85:95]
    #queryFrames = refFrames
    
    RE = recog(refFrames)
    RE.settle_activity(100, RE.net.E)
    results = np.zeros((len(refFrames), len(queryFrames)))
    for img in range(len(queryFrames)):
        E,diff = RE.step(queryFrames[img])
        results[:,img] = E
        prediction = np.argmax(E)
        if prediction == img:
            print("QUERY index [%d] SUCESSFUL PREDICTION" % img)
        else:
            #print("QUERY index [%d] FAILED OUTPUT = %d" % (img, prediction))
            pass
        plt.clf()
        plt.ylim((0,1))
        plt.stem(E, markerfmt='C0.')
        plt.pause(0.05)
    plt.clf()
    plt.ylim((0,1))
    plt.stem(E, markerfmt='C0.')
    plt.show()
    """
    while True:
        E, diff = RE.step(results[:,-1])
        plt.clf()
        plt.ylim((0,1))
        plt.stem(E, markerfmt='C0.')
        plt.pause(0.05)
    """ 
    """
    i = 0
    while True:
        try:
            frame = np.concatenate((refFrames[i], queryFrames[i]), axis=0)
        except:
            break
        frame = cv2.resize(frame, None, fx=10, fy=10,  interpolation=cv2.INTER_AREA)
        cv2.imshow('frame', frame)
        i += 1
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    """
    net = ATTRACTOR()#len(refFrames))
    NN_w = net.W_EE
    NE = net.NE
    #IN = np.zeros((100,NE))
    #IN[np.arange(100),np.repeat(np.arange(NE),10)]=1
    IN0 = np.exp(np.true_divide(-np.power(np.arange(NE) - 35,2),25)) 
    IN1 = np.ones((100,1)) * IN0
    IN0 = 2 * np.exp(np.true_divide(-np.power(np.arange(NE) - 40,2),25)) 
    IN2 = np.ones((200,1)) * IN0
    IN = np.concatenate((IN1,IN2),axis=0)
    '''
    IN = np.zeros((30,NE))
    for i in range(NE):
        IN = np.concatenate((IN,build_packet(NE,i,3,10)),axis=0)
    '''
    results = np.zeros((NE, np.shape(IN)[0]))
    print("TESTS: {}".format(np.shape(results)[1]))
    
    """
    for put in range(len(IN)):
        E, diff = net.step(IN[put])
        print("INPUT TEST {} [ERROR]:{}".format(put,np.around(diff, decimals=6)))
        results[:,put] = E
        plt.clf()
        plt.ylim((0,1))
        plt.stem(E, markerfmt='C0.')
        plt.pause(0.05)

    plt.stem(results[:,-1])
    plt.show()
    
    del refFrames[:]
    del queryFrames[:]
    """  
   
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X,Y = np.mgrid[0:NE:1, 0:np.shape(IN)[0]:1]
    surf = ax.plot_surface(X, Y, results, cmap='afmhot')
    plt.show()
    """
