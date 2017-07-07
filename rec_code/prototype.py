#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abhik
"""

import pandas
from sklearn.cross_validation import train_test_split
import numpy as np
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

#inspect = song_df.head(500)
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

global train_data
global user_id 
global item_id 
user_id = 'user_id'
item_id = 'song'

    
def get_item_users(item):
    item_data = train_data[train_data[item_id] == item]
    item_users = set(item_data[user_id].unique())
    return item_users

def get_user_items(user):
    user_data = train_data[train_data[user_id] == user]
    user_items = list(user_data[item_id].unique())
    return user_items

def get_all_items_train_data():
    all_items = list(train_data[item_id].unique())
    return all_items

all_songs = get_all_items_train_data()
user_songs = all_songs


#user_songs = get_user_items(users[4])
  
####################################
#Get users for all songs in user_songs.
####################################
user_songs_users = []        
for i in range(0, len(user_songs)):
    user_songs_users.append(get_item_users(user_songs[i]))
    
###############################################
#Initialize the item cooccurence matrix of size 
#len(user_songs) X len(songs)
###############################################
cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
   
#############################################################
#Calculate similarity between user songs and all unique songs
#in the training data
#############################################################
for i in range(0,len(all_songs)):
    #Calculate unique listeners (users) of song (item) i
    songs_i_data = train_data[train_data[item_id] == all_songs[i]]
    users_i = set(songs_i_data[user_id].unique())
    for j in range(0,len(user_songs)):       
        #Get unique listeners (users) of song (item) j
        users_j = user_songs_users[j]
        #Calculate intersection of listeners of songs i and j
        users_intersection = users_i.intersection(users_j)
        #Calculate cooccurence_matrix[i,j] as Jaccard Index
        if len(users_intersection) != 0:
            #Calculate union of listeners of songs i and j
            users_union = users_i.union(users_j)
            cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
        else:
            cooccurence_matrix[j,i] = 0

user_songs = get_user_items(users[4])
user = users[4]


user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
#Sort the indices of user_sim_scores based upon their value
#Also maintain the corresponding score
sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

#Create a dataframe from the following
columns = ['user_id', 'song', 'score', 'rank']
#index = np.arange(1) # array of numbers for the number of samples
df = pandas.DataFrame(columns=columns)
#Fill the dataframe with top 10 item based recommendations
rank = 1 
for i in range(0,len(sort_index)):
    if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
        df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
        rank = rank+1
        
if df.shape[0] == 0:
    print("The current user has no songs for training the item similarity based recommendation model.")
    print -1
else:
    print df
    
    
    
