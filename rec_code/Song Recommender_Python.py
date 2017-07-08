#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abhik
"""
"""
-------------
Dataset used:
-------------
Million Songs Dataset
Source: http://labrosa.ee.columbia.edu/millionsong/
Paper: http://ismir2011.ismir.net/papers/OS6-1.pdf

The current notebook uses a subset of the above data containing 10,000 songs obtained from:
https://github.com/turi-code/tutorials/blob/master/notebooks/recsys_rank_10K_song.ipynb

"""


import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation

"""link to datasets"""
#triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
#songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1_1 = pandas.read_csv("10000.csv")
song_df_1_2 = pandas.read_csv("20000.csv")

frames = [song_df_1_1,song_df_1_2]
song_df_1 = pandas.concat(frames)
song_df_2 =  pandas.read_csv("song_data.csv")


#Merge two dataframes to combine the songs and user 
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 

#inspection 
song_df.head()

#Length of the dataset
len(song_df)


#Create a subset as Data is large and processing might take a lot of time 
song_df = song_df.head(5000)

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name'] 

#Showing the most popular songs ( Content based recommendation)

"""song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])
"""

#Count number of unique users in the dataset
users = song_df['user_id'].unique()
len(users)

#Count Number of Unique songs in the Database 
songs = song_df['song'].unique()
len(songs)

#test train split 
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(train_data.head(5))

#Recommenders.popularity_recommender_py
#using a function from different py file and making instance of that
pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')
#using the same predictor 
user_id = users[5]
pm.recommend(user_id)
#different user 
user_id = users[8]
pm.recommend(user_id)




#Recommenders.item_similarity_recommender_py


#Create an instance of item similarity based recommender class


is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

#Personalized Approach
user_id = users[5]
user_items = is_model.get_user_items(user_id)

print("Training data songs for the user userid: %s:" % user_id)
"""for user_item in user_items:
    print(user_item)
"""
#Recommend songs for the user using personalized model
is_model.recommend(user_id)

user_id = users[7]
user_items = is_model.get_user_items(user_id)

print("Training data songs for the user userid: %s:" % user_id)
"""for user_item in user_items:
    print(user_item)
"""
is_model.recommend(user_id)



###############################################################################
is_model.get_similar_items(['U Smile - Justin Bieber'])
is_model.get_similar_items(['Yellow - Coldplay'])

###############################################################################

#Evaluation.precision_recall_calculator
#Percentage
user_sample = 0.20

"""
pr = Evaluation.precision_recall_calculator(test_data, train_data, pm, is_model)
#Precision_recall
(pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)

#Plot precision recall curve

import pylab as pl

#Method to generate precision and recall curve
def plot_precision_recall(m1_precision_list, m1_recall_list, m1_label, m2_precision_list, m2_recall_list, m2_label):
    pl.clf()    
    pl.plot(m1_recall_list, m1_precision_list, label=m1_label)
    pl.plot(m2_recall_list, m2_precision_list, label=m2_label)
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 0.20])
    pl.xlim([0.0, 0.20])
    pl.title('Precision-Recall curve')
    #pl.legend(loc="upper right")
    pl.legend(loc=9, bbox_to_anchor=(0.5, -0.2))
    pl.show()

plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")
"""



#http://antoinevastel.github.io/machine%20learning/python/2016/02/14/svd-recommender-system.html


"""
import math as mt
import csv
from sparsesvd import sparsesvd #used for matrix factorization
import numpy as np
from scipy.sparse import csc_matrix #used for sparse matrix
from scipy.sparse.linalg import * #used for matrix multiplication


#constants defining the dimensions of our User Rating Matrix (URM)
MAX_PID = 4
MAX_UID = 5

#Compute SVD of the user ratings matrix
def computeSVD(urm, K):
    U, s, Vt = sparsesvd(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(np.transpose(U), dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt

#Compute estimated rating for the test user
def computeEstimatedRatings(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 

    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        #we convert the vector to dense format in order to get the indices 
        #of the movies with the best estimated ratings 
        estimatedRatings[userTest, :] = prod.todense()
        recom = (-estimatedRatings[userTest, :]).argsort()[:250]
    return recom


# ### Use SVD to make predictions for a test user id, say 4


#Used in SVD calculation (number of latent factors)
K=2

#Initialize a sample user rating matrix
urm = np.array([[3, 1, 2, 3],[4, 3, 4, 3],[3, 2, 1, 5], [1, 6, 5, 2], [5, 0,0 , 0]])
urm = csc_matrix(urm, dtype=np.float32)

#Compute SVD of the input user ratings matrix
U, S, Vt = computeSVD(urm, K)

#Test user set as user_id 4 with ratings [0, 0, 5, 0]
uTest = [4]
print("User id for whom recommendations are needed: %d" % uTest[0])

#Get estimated rating for test user
print("Predictied ratings:")
uTest_recommended_items = computeEstimatedRatings(urm, U, S, Vt, uTest, K, True)
print(uTest_recommended_items)


(Note*: The predicted ratings by the code include the items already rated by test user as well. This has been left purposefully like this for better understanding of SVD).

SVD tutorial: http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
# ## Understanding Intuition behind SVD
SVD result gives three matrices as output: U, S and Vt (T in Vt means transpose). Matrix U represents user vectors and Matrix Vt represents item vectors. In simple terms, U represents users as 2 dimensional points in the latent vector space, and Vt represents items as 2 dimensional points in the same space.
Next, we print the matrices U, S and Vt and try to interpret them. Think how the points for users and items will look like in a 2 dimensional axis. For example, the following code plots all user vectors from the matrix U in the 2 dimensional space. Similarly, we plot all the item vectors in the same plot from the matrix Vt.



get_ipython().magic('matplotlib inline')
from pylab import *

#Plot all the users
print("Matrix Dimensions for U")
print(U.shape)

for i in range(0, U.shape[0]):
    plot(U[i,0], U[i,1], marker = "*", label="user"+str(i))

for j in range(0, Vt.T.shape[0]):
    plot(Vt.T[j,0], Vt.T[j,1], marker = 'd', label="item"+str(j))    
    
legend(loc="upper right")
title('User vectors in the Latent semantic space')
ylim([-0.7, 0.7])
xlim([-0.7, 0])
show()

"""

