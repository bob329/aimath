import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series


#定义朴素贝叶斯模型
class NBClassify(object):
    def __init__(self):
        #tabProbablity核心字典，记录各类别的先验概率，格式：{'cold':概率值, 'allergy': 概率值, 'concussion': 概率值}
        _tagProbablity=None
        #featuresProbablity核心字典，记录各类别下各特征取值的条件概率。三级字典，
        #格式：类别1: {'职业1': {'症状1': 概率值, ...'症状n': 概率值}, '职业2':{}...},类别2：{'职业1': {'症状1': 概率值, ...'症状n': 概率值},
        _featuresProbablity=None

    def train(self,df):
        # 计算每种类别的先验概率
        self._tagProbablity=df['label'].value_counts(value for value in df['label'])
        print("各类别的先验概率：\n",self._tagProbablity)

        # 计算各特征及对应取值的出现次数dictFeaturesBase
        #格式：  {特征1:{值1:出现5次, 值2:出现1次}, 特征2:{值1:出现1次, 值2:出现5次}}
        dictFeaturesBase={}.fromkeys(df.columns)
        for column in df.columns:
             seriesFeature = df[column].value_counts()
             dictFeaturesBase[column] =seriesFeature
        #从特征值字典删去类别信息
        del dictFeaturesBase['label']

        # 初始化字典 dictFeatures
        #格式：{类别1:{'特征1':{'值1':None,...'值n':None},'特征2':{...}},类别2：{'特征1':{'值1':None, ...},...}
        dictFeatures = {}.fromkeys(df['label'])
        for key in dictFeatures.keys():
            dictFeatures[key] = {}.fromkeys([key for key in dictFeaturesBase])
        for key, value in dictFeatures.items():
            for subkey in value.keys():
                value[subkey] = {}.fromkeys([x for x in dictFeaturesBase[subkey].keys()])

        # 计算各类别、对应特征及对应取值的出现次数，存入字典dictFeatures
        for i in range(0, len(df)):
            label=df.iloc[i]['label']    #类别
            for feature in columnsName[0:6]:   #对应的特征
                fvalue=df.iloc[i][feature]  #对应的特征取值
                if dictFeatures[label][feature][fvalue] == None:
                    dictFeatures[label][feature][fvalue] = 1 #该类别下该特征值第一个出现的样本
                else:
                    dictFeatures[label][feature][fvalue] +=1  #如果已有，次数加一


        # 该类数据集若未涵盖此特征值时，加入Laplace平滑项
        for tag, featuresDict in dictFeatures.items():
            for featureName, featureValueDict in featuresDict.items():
                for featureKey, featureValues in featureValueDict.items():
                    if featureValues == None:
                        featureValueDict[featureKey] = 1

        # 由字典dictFeatures计算每个类别下每种特征对应值的概率，即特征的似然概率P(feature|tag)
        for tag, featuresDict in dictFeatures.items():
            for featureName, featureValueDict in featuresDict.items():
                totalCount = sum([x for x in featureValueDict.values() if x != None])
                for featureKey, featureValues in featureValueDict.items():
                    featureValueDict[featureKey] = featureValues / totalCount
        self._featuresProbablity = dictFeatures
        print("每个类别下每种特征对应值的似然概率:\n",dictFeatures)

    # 对测试集进行预测
    def classify(self, featureTuple):
        resultDict = {}
        # 计算样本属于每个类别的后验概率
        for tag, featuresDict in self._featuresProbablity.items():
            iNumList = []
            i=0
            #将各特征值对应的似然概率添加到列表iNumList
            for feature,featureValueDict in featuresDict.items():
                featureValue=str(featureTuple[i])
                iNumList.append(self._featuresProbablity[tag][feature][featureValue])
                i=i+1
            #列表iNumList中的概率相乘，得到似然概率
            conditionProbability = 1
            for iNum in iNumList:
                conditionProbability *= iNum
            #将先验概率乘以似然概率得到后验概率resultDict
            resultDict[tag] = self._tagProbablity[tag] * conditionProbability
        # 对比每个类别的后验概率resultDict的大小
        resultList = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)
        #返回最大后验概率的类别
        return resultList[0][0]