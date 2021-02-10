#importing the libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#read csv file
df = pd.read_csv("movie_dataset.csv")

#selecting features
features = ['keywords' , 'cast' , 'genres' , 'director']

#replace Nan with empty string
for feature in features:
    df[feature] = df[feature].fillna('')

#creating a column which combines all selected features
def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        print("ERROR")

#applying combine_feature function
df["combined_feature"] = df.apply(combine_features, axis=1)

#convert title into lowercase
df['title'] = df['title'].str.lower()

#create count matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_feature"])

#compute the cosine similarity
cosine_similarity = cosine_similarity(count_matrix)
movie_user_likes = input("enter the movie you like")

#get index from title function
def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


#get title from index function
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]


#get index of movie from title
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_similarity[movie_index]))


#get list of similar movies in descending order
sorted_similar_movies = sorted(similar_movies , key=lambda x:x[1] , reverse=True)
#print title of first 50 movies
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i = i+1
    if i > 50:
        break
