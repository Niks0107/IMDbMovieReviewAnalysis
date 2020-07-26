import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from html.parser import HTMLParser
from cleantext import clean
from sklearn.metrics import accuracy_score


CurrentDir = os.path.dirname(__file__)
TrainDir = os.path.join(CurrentDir,'train')
TestDir = os.path.join(CurrentDir,'test')

'''
Reading train and test data
'''
def ReadData(train,TrainDir,TestDir):
    Data = pd.DataFrame()
    for folder in ['pos','neg']:
        if train:
            folderPath = os.path.join(TrainDir,folder)
        else:
            folderPath = os.path.join(TestDir,folder)
        os.chdir(folderPath)
        for file in os.listdir(folderPath):
            with open(file,encoding="utf8") as f:
                text = f.read()
            label = folder
            textdf = pd.DataFrame({'text':[text],'label':[label]})
            Data = pd.concat([Data,textdf],ignore_index=True)
                
    return Data

TrainData = ReadData(True,TrainDir,TestDir)
TestData = ReadData(False,TrainDir,TestDir)

'''
Cleaning the Text
'''
def cleanText(text):
    text = clean(text,no_urls=True,no_emails=True,
           no_phone_numbers=True,         # replace all phone numbers with a special token
            no_numbers=True,               # replace all numbers with a special token
            no_digits=True,                # replace all digits with a special token
            no_currency_symbols=True,      # replace all currency symbols with a special token
            no_punct=True,   )
    return text
def preprocessData(Data):
    for i in range(Data.shape[0]):
        Data.iloc[i,0] = cleanText(Data.iloc[i,0])   
    return Data

Train = preprocessData(TrainData.copy())
Test = preprocessData(TestData.copy())

'''
Model Bulding
1) tfidf vectorizer
2) Classifier
'''

xtrain,xtest,ytrain,ytest = train_test_split(Train['text'],Train['label'])

def Modelbuilding(xtrain,xtest,ytrain,ytest):
    tfidf = TfidfVectorizer(max_features=10000)
    xtrain = tfidf.fit_transform(xtrain)
    xtest = tfidf.transform(xtest)
    
    lr = LogisticRegression()
    lr.fit(xtrain,ytrain)
    preds = lr.predict(xtest)
    print("Accuracy : ",accuracy_score(ytest,preds))
    return tfidf,lr

tfidf,lr = Modelbuilding(xtrain,xtest,ytrain,ytest)

testInd = tfidf.transform(Test['text'])
TestPreds = lr.predict(testInd)
print("Accuracy : ",accuracy_score(Test['label'],TestPreds))

'''
Results : Validation Accuracy = 87.9%
          Test Accuracy = 88%  
'''




