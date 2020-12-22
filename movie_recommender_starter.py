import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# Read CSV File
df = pd.read_csv("data/movie_dataset.csv")

# Under the dataset
# df.head()
# df.info()
# Select Features and Create a column in DF which combines all selected features
features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
    df[feature] = df[feature].fillna("")


def combine_features(row):
    try:
        return row['keywords'] + " "+row['cast']+" "+row['genres']+" "+row['director']
    except:
        print("Error: ", row)


df['combined_features'] = df.apply(combine_features, axis=1)

# print(df['combined_features'].head(5))

# Create count matrix from this new combined and Compute the Cosine Similarity based on the count_matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(count_matrix)

# Test the model

movie_user_likes = "Avatar"

# Get index of this movie from its title - get the row from the matrix, enumerate it to get the index and sim score in a list of tuple and then sort using the sim score
movie_index = get_index_from_title(movie_user_likes)

# Get a list of similar movies in descending order of similarity score - get the index the enumerate on the list
similar_movies = list(enumerate(cosine_sim[movie_index]))

# sort the list of tuples based on the sim scores
sorted_similar_movies = sorted(
    similar_movies, key=lambda x: x[1], reverse=True)

# Print titles of first 50 movies
i = 0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i += 1
    if i > 50:
        break
