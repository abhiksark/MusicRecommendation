
# coding: utf-8

# # Building a song recommender
-------------
Dataset used:
-------------
Million Songs Dataset
Source: http://labrosa.ee.columbia.edu/millionsong/
Paper: http://ismir2011.ismir.net/papers/OS6-1.pdf

The current notebook uses a subset of the above data containing 10,000 songs obtained from:
https://github.com/turi-code/tutorials/blob/master/notebooks/recsys_rank_10K_song.ipynb
# In[1]:

get_ipython().magic('matplotlib inline')

import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation


# # Load music data

# In[2]:

#Read userid-songid-listen_count triplets
#This step might take time to download data from external sources
triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1 = pandas.read_table(triplets_file,header=None)
song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#Read song  metadata
song_df_2 =  pandas.read_csv(songs_metadata_file)

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 


# # Explore data
# 
# Music data shows how many times a user listened to a song, as well as the details of the song.

# In[3]:

song_df.head()


# ## Length of the dataset

# In[4]:

len(song_df)


# ## Create a subset of the dataset

# In[5]:

song_df = song_df.head(10000)

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']


# ## Showing the most popular songs in the dataset

# In[6]:

song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage']  = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count', 'song'], ascending = [0,1])


# ## Count number of unique users in the dataset

# In[8]:

users = song_df['user_id'].unique()


# In[9]:

len(users)


# ## Quiz 1. Count the number of unique songs in the dataset

# In[10]:

###Fill in the code here
songs = song_df['song'].unique()
len(songs)


# # Create a song recommender

# In[11]:

train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)
print(train_data.head(5))


# ## Simple popularity-based recommender class (Can be used as a black box)

# In[ ]:

#Recommenders.popularity_recommender_py


# ### Create an instance of popularity based recommender class

# In[12]:

pm = Recommenders.popularity_recommender_py()
pm.create(train_data, 'user_id', 'song')


# ### Use the popularity model to make some predictions

# In[13]:

user_id = users[5]
pm.recommend(user_id)


# ### Quiz 2: Use the popularity based model to make predictions for the following user id (Note the difference in recommendations from the first user id).

# In[14]:

###Fill in the code here
user_id = users[8]
pm.recommend(user_id)


# ## Build a song recommender with personalization
# 
# We now create an item similarity based collaborative filtering model that allows us to make personalized recommendations to each user. 

# ## Class for an item similarity based personalized recommender system (Can be used as a black box)

# In[ ]:

#Recommenders.item_similarity_recommender_py


# ### Create an instance of item similarity based recommender class

# In[15]:

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')


# ### Use the personalized model to make some song recommendations

# In[16]:

#Print the songs for the user in training data
user_id = users[5]
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)


# ### Quiz 3. Use the personalized model to make recommendations for the following user id. (Note the difference in recommendations from the first user id.)

# In[17]:

user_id = users[7]
#Fill in the code here
user_items = is_model.get_user_items(user_id)
#
print("------------------------------------------------------------------------------------")
print("Training data songs for the user userid: %s:" % user_id)
print("------------------------------------------------------------------------------------")

for user_item in user_items:
    print(user_item)

print("----------------------------------------------------------------------")
print("Recommendation process going on:")
print("----------------------------------------------------------------------")

#Recommend songs for the user using personalized model
is_model.recommend(user_id)


# ### We can also apply the model to find similar songs to any song in the dataset

# In[18]:

is_model.get_similar_items(['U Smile - Justin Bieber'])


# ### Quiz 4. Use the personalized recommender model to get similar songs for the following song.

# In[19]:

song = 'Yellow - Coldplay'
###Fill in the code here
is_model.get_similar_items([song])


# # Quantitative comparison between the models
# 
# We now formally compare the popularity and the personalized models using precision-recall curves. 

# ## Class to calculate precision and recall (This can be used as a black box)

# In[20]:

#Evaluation.precision_recall_calculator


# ## Use the above precision recall calculator class to calculate the evaluation measures

# In[20]:

start = time.time()

#Define what percentage of users to use for precision recall calculation
user_sample = 0.05

#Instantiate the precision_recall_calculator class
pr = Evaluation.precision_recall_calculator(test_data, train_data, pm, is_model)

#Call method to calculate precision and recall values
(pm_avg_precision_list, pm_avg_recall_list, ism_avg_precision_list, ism_avg_recall_list) = pr.calculate_measures(user_sample)

end = time.time()
print(end - start)


# ## Code to plot precision recall curve

# In[21]:

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


# In[22]:

print("Plotting precision recall curves.")

plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")


# ### Generate Precision Recall curve using pickled results on a larger data subset(Python 3)

# In[23]:

print("Plotting precision recall curves for a larger subset of data (100,000 rows) (user sample = 0.005).")

#Read the persisted files 
pm_avg_precision_list = joblib.load('pm_avg_precision_list_3.pkl')
pm_avg_recall_list = joblib.load('pm_avg_recall_list_3.pkl')
ism_avg_precision_list = joblib.load('ism_avg_precision_list_3.pkl')
ism_avg_recall_list = joblib.load('ism_avg_recall_list_3.pkl')

print("Plotting precision recall curves.")
plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")


# ### Generate Precision Recall curve using pickled results on a larger data subset(Python 2.7)

# In[24]:

print("Plotting precision recall curves for a larger subset of data (100,000 rows) (user sample = 0.005).")

pm_avg_precision_list = joblib.load('pm_avg_precision_list_2.pkl')
pm_avg_recall_list = joblib.load('pm_avg_recall_list_2.pkl')
ism_avg_precision_list = joblib.load('ism_avg_precision_list_2.pkl')
ism_avg_recall_list = joblib.load('ism_avg_recall_list_2.pkl')

print("Plotting precision recall curves.")
plot_precision_recall(pm_avg_precision_list, pm_avg_recall_list, "popularity_model",
                      ism_avg_precision_list, ism_avg_recall_list, "item_similarity_model")


# The curve shows that the personalized model provides much better performance over the popularity model. 

# # Matrix Factorization based Recommender System
Using SVD matrix factorization based collaborative filtering recommender system
--------------------------------------------------------------------------------

The following code implements a Singular Value Decomposition (SVD) based matrix factorization collaborative filtering recommender system. The user ratings matrix used is a small matrix as follows:

        Item0   Item1   Item2   Item3
User0     3        1       2      3
User1     4        3       4      3
User2     3        2       1      5
User3     1        6       5      2
User4     0        0       5      0

As we can see in the above matrix, all users except user 4 rate all items. The code calculates predicted recommendations for user 4.
# ### Import the required libraries

# In[25]:

#Code source written with help from: 
#http://antoinevastel.github.io/machine%20learning/python/2016/02/14/svd-recommender-system.html

import math as mt
import csv
from sparsesvd import sparsesvd #used for matrix factorization
import numpy as np
from scipy.sparse import csc_matrix #used for sparse matrix
from scipy.sparse.linalg import * #used for matrix multiplication

#Note: You may need to install the library sparsesvd. Documentation for 
#sparsesvd method can be found here:
#https://pypi.python.org/pypi/sparsesvd/


# ### Methods to compute SVD and recommendations

# In[26]:

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

# In[27]:

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


# ### Quiz 4
a.) Change the input matrix row for test userid 4 in the user ratings matrix to the following value. Note the difference in predicted recommendations in this case.

i.) [5 0 0 0]


(Note*: The predicted ratings by the code include the items already rated by test user as well. This has been left purposefully like this for better understanding of SVD).

SVD tutorial: http://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm
# ## Understanding Intuition behind SVD
SVD result gives three matrices as output: U, S and Vt (T in Vt means transpose). Matrix U represents user vectors and Matrix Vt represents item vectors. In simple terms, U represents users as 2 dimensional points in the latent vector space, and Vt represents items as 2 dimensional points in the same space.
Next, we print the matrices U, S and Vt and try to interpret them. Think how the points for users and items will look like in a 2 dimensional axis. For example, the following code plots all user vectors from the matrix U in the 2 dimensional space. Similarly, we plot all the item vectors in the same plot from the matrix Vt.

# In[28]:

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

