#-*- coding:utf-8 -*-
'''实现ID3决策树算法'''
from math import log
import operator
from treePlotter import *
'''计算给定数据集的香农熵'''
def calcShannonEnt(dataSet):
    numEntries = len (dataSet)
    labelCounts= {}
    for featVec in dataSet:
        '''为所有可能分类创建字典'''
        currentLabel= featVec[-1]    #当前样本的类标
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] +=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries    #p(xi)*log2p(xi)
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

'''创建数据集'''
def createDataSet(filename):
    fr=open(filename)
    instances=[inst.strip().split('\t') for inst in fr.readlines()]
    return instances
    #lensesTree=trees.createTree(lenses,lensesLabels)
def createDataSet2():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

'''根据给定的特征和特征值划分数据集'''
def splitDataSet(dataSet,featureID, value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[featureID]==value:
            reducedFeatVec=featVec[:featureID]
            reducedFeatVec.extend(featVec[featureID+1:])    #去掉样本中的该划分特征值形成新的样本
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''选择最好的数据集划分方式'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures= len(dataSet[0]) -1    #特征数
    baseEntropy=calcShannonEnt(dataSet)    #未划分前的信息熵
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):         #用每个特征对数据集进行一次划分，选择划分效果最好的特征
        featList=[example[i] for example in dataSet]     #由数据集在该特征(下标为i)中的所有特征值组成一个列表
        uniqueVals=set(featList)        #该特征的所有可能值
        newEntropy=0.0
        for value in uniqueVals:                 #计算使用当前特征进行划分时的信息熵
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if (infoGain>bestInfoGain):        #选择最好的信息增益
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

'''当所有的属性都使用了之后，若类标签仍然不是唯一的，就采用多数表决判断叶子结点的分类'''
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote]=0
        classCount[vote] +=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

'''创建决策树'''
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):            #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) ==1:          #由于从树根往下划分时，每划分一次样本的特征就少一个，若最后只剩一个特征时表示无法简单地返回唯一的类标，需要多数表决
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])   #删除已使用过的属性
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

'''使用决策树进行分类'''
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

'''使用pickle模块存储决策树'''
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):       #使用pickle从文件加载对象
    import pickle
    fr=open(filename,'rb')
    return pickle.load(fr)

def main():
    dataFileName='lenses.txt'      #数据集文件
    storedTreeFile='storedTree.txt'    #序列化存储所在文件
    instances=createDataSet(dataFileName)
    labels=['age','prescript','astigmatic','tearRate']
    tempLabels=labels[:]
    createdTree=createTree(instances,tempLabels)      #构造树过程会修改labels列表
    storeTree(createdTree,storedTreeFile)
    print("The created Decision Tree:\n",createdTree)
    #test(storedTreeFile,labels)
    test2(storedTreeFile,labels,dataFileName)
    #createPlot(createdTree)  #绘制决策树图
    
def test(storedTreeFile,labels):                      #使用一个用例测试训练后的模型
    trainingTree=grabTree(storedTreeFile)    #获取训练后的树模型
    testInstance=['pre','myope','no','reduced']   #测试用例
    classifiedLabel=classify(trainingTree,labels,testInstance)
    print("The ClassifiedLabel is:",classifiedLabel)
    
def test2(storedTreeFile,labels,testSetFile):        #使用测试数据集文件测试训练后的模型
    trainingTree=grabTree(storedTreeFile)
    fr=open(testSetFile,'r')
    correctNum=0      #正确分类的用例数
    totalNum=0          #测试集中总共的用例数
    for oneLine in fr.readlines():
        instance=oneLine.strip().split('\t')
        totalNum+=1
        tempList=instance[:-1]
        targetLabel=instance[-1]
        if targetLabel==classify(trainingTree,labels,tempList):
            correctNum+=1
    print("The classifying accruracy:",correctNum/float(totalNum))

if __name__=='__main__':
    main()