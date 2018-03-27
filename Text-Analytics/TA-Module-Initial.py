#Import libraries for use
import pandas  as  pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import codecs
import glob


from wordcloud import STOPWORDS, wordCloud


#Functions:


#Get inputs from user:
#Input dataset
rawData = input("Enter filename (eg. ): ")

#Input relevant column_name:
column_name = input("Enter list of comment columns:  ")


################################# BUILD-CORPUS #################################

#Build Corpus
def builCorpus(data, filename):
    cwd = os.getcwd()
    os.mkdrir('Corpus')
    os.chdir(cwd + '/Corpus')

    cleanedData = data.str.lower().drop(how='any')
    file = codecs.open(filename, "w", "utf-8")

    for comment in data:
        file.write(comment)
        file.write('\n')

    file.close()
    return

################################## WORD-COUNT ##################################

#Build WordCount
def buildWordCount(filename):
    cwd = os.getcwd()
    path = cwd + '/Corpus/'
    file = open(path + filename, 'r')
    counter = {}

    for word in file.read().split():
        if word not in counter:
            counter[word] = 1
        else:
            counter[word] += 1
    return

    counterFrame = pd.Series(counter).to_frame()
    counterFrame.columns = ['Count']
    counterFrame = counterFrame[counterFrame['Count'] > counterFrame['Count'].mean()]
    counterFrame = counterFrame[~counterFrame.isin(STOPWORDS)]

    return counterFrame.sort_values('Count', ascending=False)

################################## WORD-CLOUD ##################################

#Build WordCloud
def buildWordCloud(data):
    ...
    return

def buildPercentMentions(data):
    ...
    return

#################################### TF-IDF ####################################