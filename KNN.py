from numpy import *
import operator
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sys



import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import datasets

import random
import math
import operator

import threading
import time
from threading import Thread


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)-1):
            for y in range(8):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def loadDataset2(filename, trainSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)):
            for y in range(784):
                dataset[x][y] = float(dataset[x][y])
            trainSet.append(dataset[x])


def loadDataset3(filename, testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset)):
            for y in range(784):
                dataset[x][y] = float(dataset[x][y])
            testSet.append(dataset[x])

def loadDatasetforaccuracy(filename, split,trainingSet=[],testSet=[]):
    actual=[]
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range (1,len(dataset)-1):
            for y in range(785):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                    trainingSet.append(dataset[x])
            else:
                    testSet.append(dataset[x])




def euclideanDistance(instance1, instance2, length):
    distance = 0
    assert isinstance(length, int)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)


def manhattanDistance(instance1,instance2,length):
    distance =0
    assert isinstance(length,int)
    for x in range(length):
        distance += abs(instance1[x]-instance2[x])
    return distance


def l3Distance(instance1,instance2,length):
    distance =[]
    assert isinstance(length,int)
    for x in range(length):
        distance.append( abs(instance1[x]-instance2[x]))
    return max(distance)




def getNeigthbors(trainingSet, testInstance, k):
    distances =[]
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):

       # dist = euclideanDistance(testInstance,trainingSet[x],length)
       # dist = manhattanDistance(testInstance, trainingSet[x], length)
        dist = l3Distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet,predictions):
    truepos = 0
    trueneg=0
    falsepos=0
    falseneg=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            if testSet[x][-1]=='0' :
                trueneg += 1
            else:
                truepos += 1

        if testSet[x][-1] != predictions[x]:
            if testSet[x][-1]=='1' :
                falsepos+=1
            else:
                falseneg+=1
    df = pd.DataFrame({'true': [truepos, trueneg], 'false': [falsepos, falseneg]}, index=['pos', 'neg'])
    print(df)
    print(truepos,trueneg,falseneg,falsepos)
    return ((trueneg+truepos)/float(len(testSet))) * 100.00



def getAccuracy3(testSet,predictions):
    truepos = 0
    trueneg=0
    falsepos=0
    falseneg=0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            if testSet[x][-1]==2 :
                truepos += 1
            else:
                trueneg += 1

        if testSet[x][-1] != predictions[x]:
            if testSet[x][-1]==2 :
                falsepos+=1
            else:
                falseneg+=1
    df = pd.DataFrame({'true': [truepos, trueneg], 'false': [falsepos, falseneg]}, index=['pos', 'neg'])
    print(df)
    print(truepos,trueneg,falseneg,falsepos)
    return ((trueneg+truepos)/float(len(testSet))) * 100.00



def getAccuracy2(testSet, predictions):
            correct = 0
            for x in range(len(testSet)):
                if testSet[x][-1] == predictions[x]:
                    correct += 1
            return (correct / float(len(testSet))) * 100.00



class MyThread(Thread):
    def __init__(self,trainingSet=[],testSet=0,k=0):
        Thread.__init__(self)
        self.k = k
        self.testSet = testSet
        self.trainingSet = trainingSet
        
    def run(self):
        self.result=getNeigthbors(self.trainingSet,self.testSet,self.k)
    def getresult(self):
        return self.result





def main():
    start = time.clock()

    # prepare data
    trainingSet = []
    testSet = []
    split = 0.8
    trainingSet2=[]
    testSet2=[]
    trainingsetforaccuracy=[]
    testsetforaccuracy=[]
    loadDataset('diabetes.csv', split, trainingSet, testSet)

  #  loadDataset2('train.csv',trainingSet2)
  #  loadDataset3('test.csv',testSet2)

   # loadDatasetforaccuracy('train.csv',split,trainingsetforaccuracy,testsetforaccuracy)

    print 'Train Set' + repr(len(trainingSet))
    print 'Test Set' + repr(len(testSet))
  #  print 'Train Set' + repr(len(trainingSet2))
  #  print 'Test Set' + repr(len(testSet2))
  #  print 'Train Set' + repr(len(trainingsetforaccuracy))
  #  print 'Test Set' + repr(len(testsetforaccuracy))
    # generate predicitions
    predicitions = []
    predicitions2= []
    predicitions3 = []
    predicitionsforaccuracy=[]
    neighbors2=[]
    result2=[]
    k = 3

    '''
    for x in range(len(testSet2)):
         neighbors = getNeigthbors(trainingSet2,testSet2[x],k)
         print(28000-x)
         result = getResponse(neighbors)  # type: object
         predicitions2.append(result)



    with open('sample_submission.csv','w') as csvfile:
       writer =csv.writer(csvfile)
       writer.writerow(['ImageID','Label'])
       for i in range(len(predicitions2)):

           writer.writerow([i,predicitions2[i]])
    '''
    for x in range(len(testSet)):
        neighbors = getNeigthbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)  # type: object
        predicitions.append(result)
        print('> predicition = ' + repr(result) + ', actual = ' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predicitions)
    print('Accuracy:' + repr(accuracy) + '%')
    print(len(predicitions))
   # print('> predicition = '+ repr(result) + ', actual = '+ repr(testSet2[x][-1]))
   # accuracy = getAccuracy(testSet2, predicitions2)
   # print('Accuracy:' + repr(accuracy) + '%')
    #print(len(predicitions2))


    '''
    for x in range(len(testsetforaccuracy)):
        neighbors = getNeigthbors(trainingsetforaccuracy,testsetforaccuracy[x],k)
        print(len(testsetforaccuracy)-x)
        result = getResponse(neighbors)  # type: object
        predicitionsforaccuracy.append(result)
       # print('> predicition = '+ repr(result) + ', actual = '+ repr(testSet[x][-1]))
    accuracy = getAccuracy3(testsetforaccuracy, predicitionsforaccuracy)
    print('Accuracy:' + repr(accuracy) + '%')
    print(len(predicitions))
    
 
    '''
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
main()

'''
 
'''