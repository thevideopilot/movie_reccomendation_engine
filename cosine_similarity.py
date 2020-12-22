from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ['London Paris London', 'Paris Paris London']
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)
vect = count_matrix.toarray()


print(cosine_similarity(vect))
