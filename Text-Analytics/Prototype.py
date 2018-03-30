import pandas as pd
import numpy as np
import nltk
from wordcloud import WordCloud, STOPWORDS
import os
import TextAnalytics
import PreProcessor
import glob
import codecs
import matplotlib.pyplot as plt
%matplotlib inline
import shutil
import sys

import importlib 
importlib.reload(TextAnalytics)

#STEP-1: Load data and explore missing values
data_raw = pd.read_csv('cs_responses_export_2018-03-31_235959.csv', low_memory=False)
missing_raw = PreProcessor.missingSummary(data_raw)
missing_raw

#STEP-2: Clean data with col and row cutoffs
clean_df, missing_clean = PreProcessor.cleanData(data_raw, missing_raw, 75, 20)
missing_clean

clean_CAT_df['CC Segment'].unique()

#STEP-3: No Imputation neccessary -- Categorize features for PR AND TA
target = list(['OSAT Experience'])
features_numeric = list(['Client Effort', 'Professionalism','Knowledge', 'Communicating Clearly','Understanding Needs'])
features_TA = 'bac_cs_all_keymetric_employee_comment_cmt'
clean_CAT_df = PreProcessor.pr_categorize(clean_df, features_numeric)
clean_CAT_df = PreProcessor.pr_categorize(clean_CAT_df, target)

#Filter and separte Segments & OSAT-CAT
Retail_UN_PROMOTER = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Retail') & (clean_CAT_df['Understanding Needs-CAT'] == '1.Promoter'), [features_TA]]
Retail_UN_DETRACTOR = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Retail') & (clean_CAT_df['Understanding Needs-CAT'] == '2.Detractor'), [features_TA]]
Preferred_UN_PROMOTER = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Preferred') & (clean_CAT_df['Understanding Needs-CAT'] == '1.Promoter'), [features_TA]]
Preferred_UN_DETRACTOR = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Preferred') & (clean_CAT_df['Understanding Needs-CAT'] == '2.Detractor'), [features_TA]]
Fraud_UN_PROMOTER = clean_CAT_df.loc[((clean_CAT_df['CC Segment'] == 'Retail Fraud Servicing') | (clean_CAT_df['CC Segment'] == 'Preferred Small Business Fraud'))
 & (clean_CAT_df['Understanding Needs-CAT'] == '1.Promoter'), [features_TA]]
Fraud_UN_DETRACTOR = clean_CAT_df.loc[((clean_CAT_df['CC Segment'] == 'Retail Fraud Servicing') | (clean_CAT_df['CC Segment'] == 'Preferred Small Business Fraud'))
 & (clean_CAT_df['Understanding Needs-CAT'] == '2.Detractor'), [features_TA]]
Billing_UN_PROMOTER = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Billing Disputes') & (clean_CAT_df['Understanding Needs-CAT'] == '1.Promoter'), [features_TA]]
Billing_UN_DETRACTOR = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Billing Disputes') & (clean_CAT_df['Understanding Needs-CAT'] == '2.Detractor'), [features_TA]]
SMBServicing_UN_PROMOTER = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Small Business') & (clean_CAT_df['Understanding Needs-CAT'] == '1.Promoter'), [features_TA]]
SMBServicing_UN_DETRACTOR = clean_CAT_df.loc[(clean_CAT_df['CC Segment'] == 'Small Business') & (clean_CAT_df['Understanding Needs-CAT'] == '2.Detractor'), [features_TA]]

#STEP-4 Build Corpus
TextAnalytics.buildCorpus(Retail_UN_PROMOTER[features_TA], 'Corpus_Retail_PROMOTER.txt')
TextAnalytics.buildCorpus(Retail_UN_DETRACTOR[features_TA], 'Corpus_Retail_DETRACTOR.txt')
TextAnalytics.buildCorpus(Preferred_UN_PROMOTER[features_TA], 'Corpus_Preferred_PROMOTER.txt')
TextAnalytics.buildCorpus(Preferred_UN_DETRACTOR[features_TA], 'Corpus_Preferred_DETRACTOR.txt')
TextAnalytics.buildCorpus(Fraud_UN_PROMOTER[features_TA], 'Corpus_Fraud_PROMOTER.txt')
TextAnalytics.buildCorpus(Fraud_UN_DETRACTOR[features_TA], 'Corpus_Fraud_DETRACTOR.txt')
TextAnalytics.buildCorpus(Billing_UN_PROMOTER[features_TA], 'Corpus_Billing_PROMOTER.txt')
TextAnalytics.buildCorpus(Billing_UN_DETRACTOR[features_TA], 'Corpus_Billing_DETRACTOR.txt')
TextAnalytics.buildCorpus(SMBServicing_UN_PROMOTER[features_TA], 'Corpus_SMBServicing_PROMOTER.txt')
TextAnalytics.buildCorpus(SMBServicing_UN_DETRACTOR[features_TA], 'Corpus_SMBServicing_DETRACTOR.txt')

#STEP-5 Get a list of all Corpus
Corpus_list = []
for file in glob.glob('Corpus*'):
    Corpus_list.append(file)

#STEP-6 Simple WordCloud (No POS-Tagging)
for corpus in Corpus_list:
    buildWordCloud(corpus)

token = TextAnalytics.word2token('Corpus_Retail_PROMOTER.txt')
posTag = TextAnalytics.token2PosTag(token)
Nouns = TextAnalytics.findtags('NN', posTag, 25)
Adj = TextAnalytics.findtags('JJ', posTag, 25)
Nouns = Nouns['NN']
adj_dict = Adj['JJ']

Nouns
check = {word: count for word, count in Nouns}
check2 = {word: count for word, count in adj_dict}
check.update(check2)
check
wc = WordCloud().generate_from_frequencies(check)
wc.to_file('check.png')

sorted(check)

#Move all files into Images Folder
path = os.getcwd()
png_file = glob.glob('*.png')
for file in png_file:
    shutil.move(path+'/'+file, path+'/Images/'+file)

#STEP-7 Build a list of Tokens
token_list = [token_Retail_PROMOTER, token_Retail_DETRACTOR, token_Preferred_PROMOTER, token_Preferred_DETRACTOR, token_Fraud_PROMOTER, token_Fraud_DETRACTOR,
    token_Billing_PROMOTER, token_Billing_DETRACTOR, token_SMBServicing_PROMOTER, token_SMBServicing_DETRACTOR]


token_Retail_PROMOTER = TextAnalytics.word2token('Corpus_Retail_PROMOTER.txt')


#-------------------------------------------------------------------------------
check_file = codecs.open('Corpus_Billing_DETRACTOR.txt', 'r', 'utf-8').read()
check_token = nltk.word_tokenize(check_file)
importlib.reload(TextAnalytics)

len(check_token)
check_token = [word for word in check_token if len(word) > 1]
check_token = [word for word in check_token if not word.isnumeric()]
check_token = [word.lower() for word in check_token]

default_stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords_file = 'stopwords.txt'
custom_stopwords = set(codecs.open(stopwords_file, 'r', 'utf-8').read().splitlines())
all_stopwords = default_stopwords | custom_stopwords

check_token = [word for word in check_token if word not in all_stopwords]
fdist = nltk.FreqDist(check_token)
fdist.keys
for word, freqency in fdist.most_common(50):
    print(u'{};{}'.format(word, frequency))

check_pos = nltk.pos_tag(check_token)
check_pos_NN = TextAnalytics.findtags('NN', check_pos, 50)
check_pos_ADJ = TextAnalytics.findtags('JJ', check_pos, 50)
check_pos_NN = check_pos_NN['NN']
check_pos_ADJ = check
check_pos_ADJ_dict = {word: count for word, count in check_pos_ADJ}
check_pos_NN_dict

checkMod_token = TextAnalytics.word2token('Corpus_Billing_DETRACTOR.txt')
len(checkMod_token)
checkMod_pos = TextAnalytics.token2PosTag(checkMod_token)
checkMod_tags = TextAnalytics.findtags('NN', checkMod_pos, 50)
checkMod_tags
text =  check_file.concordance('numbers')

f=codecs.open('Corpus_Billing_DETRACTOR.txt','r','utf-8')
raw=f.read()
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)
text.to_file('x.txt')
#-------------------------------------------------------------------------------

clean_df_2 = clean_df.copy()
len(clean_df_2)
clean_df_2, missing_clean_2 = PreProcessor.cleanData(clean_df, missing_clean, 100, 65)
clean_df['Hold_Mentions'] = 0

mod_index1 = list(clean_df_2[clean_df_2[features_TA].str.contains("hold on")].index)


clean_df_2.at[mod_index1, 'Hold_Mentions'] = 1
clean_df_2.Hold_Mentions.value_counts()

col_name = "hold on".split()[0]
col_name

clean_df_2, percent_account = TextAnalytics.tagMentions(clean_df_2, features_TA, 'hold on')
percent_account
