#Import libraries for use
import os
import codecs
import glob
import pandas  as  pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import STOPWORDS, WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords

# NLTK's default English stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))

#custom_stopwords
stopwords_file = 'stopwords.txt'
custom_stopwords = set(codecs.open(stopwords_file, 'r', 'utf-8').read().splitlines())

all_stopwords = default_stopwords | custom_stopwords



#Get inputs from user:
#Input dataset
# rawData = input("Enter filename (eg. ): ")
#
# #Input relevant column_name:
# column_name = input("Enter list of comment columns:  ")


################################# BUILD-CORPUS #################################

#Build Corpus
def buildCorpus(data, filename):
    cwd = os.getcwd()

    cleanedData = data.str.lower().dropna(how='any')
    file = codecs.open(filename, "w", "utf-8")

    for comment in cleanedData:
        file.write(comment)
        file.write('\n')

    file.close()
    return

################################## WORD-COUNT ##################################

#Build WordCount
def buildWordCount(filename):
    cwd = os.getcwd()
    path = cwd
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
    file = open(data).read()
    STOPWORDS.add('bank')
    STOPWORDS.add('america')
    STOPWORDS.add('boa')
    STOPWORDS.add('associate')
    STOPWORDS.add('person')
    STOPWORDS.add('thank')
    STOPWORDS.add('account')
    STOPWORDS.add('one')
    STOPWORDS.add('problem')
    STOPWORDS.add('issue')
    STOPWORDS.add('great')
    STOPWORDS.add('branch')
    STOPWORDS.add('teller')
    STOPWORDS.add('service')
    STOPWORDS.add('told')
    STOPWORDS.add('need*')
    STOPWORDS.add('help')
    STOPWORDS.add('customer')
    STOPWORDS.add('needed')

    if data in glob.glob("*PROMOTER.txt"):
        color="Greens"
    elif data in glob.glob("*PASSIVE.txt"):
        color="Blues"
    else:
        color="Reds"

    wc = WordCloud(max_words=1000, stopwords=STOPWORDS, margin=10, background_color='white', colormap=color,
                   random_state=1).generate(file)
    # store default colored image
    default_colors = wc.to_array()
    # plt.title("Custom colors")
    # plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
    #            interpolation="bilinear")
    wc.to_file(data+'.png')
    plt.axis("off")
    plt.figure(figsize=(12,12))
    # plt.title("Default colors")
    plt.imshow(default_colors, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return

def tagMentions(data, features_TA, substring):
    col_name = substring.split()[0]
    data[col_name] = 0

    index_of_mentions = list(data[data[features_TA].str.contains(substring)].index)
    data.at[index_of_mentions, col_name] = 1
    percent_mentions = percentMentions(data[col_name])
    return data, round(percent_mentions * 100, 2)

def percentMentions(data):
    percent = data.sum() / len(data)
    return percent

##################################### NLTK #####################################

#Covert Textfile into tokens (i.e. Wordcounts)
def word2token(filename):
    file = codecs.open(filename, 'r','utf-8')
    readfile = file.read()
    token = nltk.word_tokenize(readfile)
    file.close()

    # Remove single-character tokens (mostly punctuation)
    token = [word for word in token if len(word) > 1]

    # Remove numbers
    token = [word for word in token if not word.isnumeric()]

    # # Lowercase all words (default_stopwords are lowercase too)
    token = [word.lower() for word in token]

    # Remove stopwords
    default_stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords_file = 'stopwords.txt'
    custom_stopwords = set(codecs.open(stopwords_file, 'r', 'utf-8').read().splitlines())
    all_stopwords = default_stopwords | custom_stopwords
    token = [word for word in token if word not in all_stopwords]
    return token

def token2PosTag(token):
    pos_tagged = nltk.pos_tag(token) #returns a tuple
    return pos_tagged

def findtags(tag_prefix, tagged_text, numTopWords):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
                                  if tag.startswith(tag_prefix))
    tagged_count = dict((tag, cfd[tag].most_common(numTopWords)) for tag in cfd.conditions())
    tagged_count = tagged_count[tag_prefix]
    tagged_count_dict = {word: count for word, count in tagged_count}
    return tagged_count_dict
