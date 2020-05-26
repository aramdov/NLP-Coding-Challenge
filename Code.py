  
### Coding challenge Construction data science

import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import nltk as nltk
import string
string.punctuation

from nltk.corpus import stopwords
ignore_words = stopwords.words('english')

import os 
os.getcwd()

# Data briefing
dt = pd.read_csv('City.csv')
print(dt.iloc[:3])

# Parse data into list of tuples, drop unnecessary categorical columns
# keep OBJECTID as primary key attribute and description 

def pipeline_df(dt):
    data = []
    # remove NA values and general cleanup
    dt_new = dt.dropna(subset=['OBJECTID', 'description'])
    
    for i, row in dt_new.iterrows():
        # create new column names
        id_city = row['OBJECTID']
        # lowercase all text, remove punctuations, remove white space
        ds_city = row['description'].lower().translate(str.maketrans('','', string.punctuation)).split(' ')
        # create new object containing no stopwords, aka remove stopwords from data
        ds_city_clean = [s for s in ds_city if s not in ignore_words]
        data.append((id_city, ds_city_clean))
        
    # Retrieve all unique words to be used as features
    words_u, words_ct, words = get_features(data)
    
    # Bag of words Matrix (M entries/rows and N features)
    X = build_feature_vector(data, words_u)
    
    # Return Matrix
    return X, words_u


# Return list of unique words aka our features
def get_features(data):
    all_words = np.array([])
    
    for entry in data:
        all_words = np.concatenate((all_words, entry[1]))    # Combine/Concatenate array
        all_words_u, all_words_count = np.unique(all_words, return_counts=True) # remove duplicates
    return all_words_u, all_words_count, all_words

def build_feature_vector(data, f_vector):
    M,N = len(data), len(f_vector)     #create matrix, M is row length, N is column length
    X = np.zeros((M,N))               # start with matrix of 0 values
    for ii, entry in enumerate(data):    # enumerte makes counter for each entry/word in description
        bag_vector = np.zeros(len(f_vector))
        for word_entry in entry[1]:     # for each word in every description
            for i, word_f in enumerate(f_vector):
                if word_entry == word_f:
                    bag_vector[i] = bag_vector[i] + 1
        X[ii,:] = bag_vector
    return X

# Input data into pipeline and get output matrix
X_tmp,features = pipeline_df(dt)

# Remove bad features Columns
X = X_tmp[:,1:]
features = features[1:]
        
X_dt = pd.DataFrame(X,columns=features)            
X_dt.head
    
# X_dt is our dataframe indexing descriptions in order for rows and the columns
# are our count on how many times the unique word appears in the respective description
# numerical values are also in the list of features


### Plotting barplots of the frequency of key words we're interested in to help us
# evaluate what the best model might be for the task at hand and move from there.
# Barplots are a way to visualize the text data after it has been quantified for this problem
# it might give us insight or an idea of what model and direction to head toward after observing
# frequencies, patterns, distributions, etc. Can also use other statistical/math plots to visualize the data

X_dt['residential'].value_counts()[1:].plot(kind='bar', title='residential')  
# If projects contain residential, odds are they will be dropped for relevancy
# becaue the Proability(Residential | Description contains Residential) ~ 1 is a safe assuumption for this sake

X_dt['large' and 'development'].value_counts()[1:].plot(kind='bar', title='Large and Development')
X_dt['large'].value_counts()[1:].plot(kind='bar', title='large')

intfeat = [int(i) for i in features if i.isdigit()]     # extract numbers from features to check distribution of square footages
mpl.hist(intfeat,range=(50000, 1000000), bins = 30)     # most numbers > 50,000 are distributed around less than 200,000
# these numbers are most likely the square footage for the project
mpl.hist(intfeat,range=(50000, 200000), bins = 30)

# Could comeup with more creative methods of analyzing the frequency of the words
# with a more complicated model, trying to identifying what words could indicate large-scale development
# like zoning, permits, etc, but for this sake we can use the frequency of the words to
# weight the conditions given for the contractor and use the weights to give out a relevancy score
# The word Scale never appeared after the filtering so can't search the frequency of that
# best I could see is large and development as maybe an indicator of large-scale development




### A metric I came up for giving a relevancy score on each project description
# is to give a value between [0,10], 10 indicating most relevant, 0 indicating not relevant

# If a description contains the word residential, it get classified as not relevant
# and gets a relevancy score of 0, I think it's a reasonable assumption that if the description contains
# the word residential, the construction project is a residential one, which the contractor does not do

# If the description contains the words large and development, it gets 3 points of relevancy score
# it's difficult and tricky to pick how much to weight the conditions for this problem at hand
# so the numbers are just a gut feeling of an appropiate weight, they are probabily inaccurate for extensive modeling
 
# If a description contains a number between 50,000 and 10,000,000 it gets points of relevancy score
# based on how high the number is, this number I assume is the square footage and I believe it is
# a safe assumption, while I could have checked the condition for a numeric value in that range and also
# containg some variation of "sqft", there are just too many ways sqft is written and hard coding the 
# variations is not worth it, I could use things like Jaccard's similarity or shingling methods to help but
# I believe this is overthinking the problem which I admit I fell victim to. To determine the relevancy score
# Contribution to relevancy score is sqft / 15,000, the denominator of this fraction depends on
# what the SF city constititues the threshold of sqft for large-scale development, information I don't know
# This means that if sqft > something is considered large-scale development, the denominator depends on that.


X_dt_matrix = X_dt.values
rel_score = []
        
for row in X_dt_matrix:
    
    score = 0 # temp score for each row, start as 0
    words = [] # list of all words that appear in a row
    
    for ii,col in enumerate(row):
        
        if col > 0.0:
            # check to see if that word appears (i.e. value that is >0)
            
            # add to our list of words
            words.append(features[ii])
            
    # Relevancy Score creation
    if 'large' in words or 'development' in words:
        score = score + 1
    
    if 'large' in words and 'development' in words:
        score = score + 3
        
    if 'residential' in words:
        score = score*0
    for word in words:
        
        if word.isdigit():
            
            num = float(word)
            if num > 50000 and num < 10000000:
                score = score + num/15000
               
    rel_score.append(score)
    
    
dt_temp = dt.dropna(subset=['OBJECTID', 'description'])
dt_temp = dt_temp.loc[:, ['OBJECTID', 'description']]       
MVPdt = dt_temp 
MVPdt['Relevance_Score'] = rel_score  # combine relevance score into dataframe with ID and description
# this is the minimum viable product
# There are some outliers in the relevance score but that's not a big problem as what's more important
# is that we tried to scale the relevance to be out of 10, probles with the metric of the sqft cause the scores
# to be high due to how points were calculated for sqft.

