from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pickle
import pandas as pd
import numpy as np
import time

start = time.time()

movie_reviews = pd.read_csv("datasets\IMDB_dataset_clean.csv")
movie_reviews.isnull().values.any()

instance = "Basically there a family where little boy Jake thinks there a zombie in his closet his parents are " \
           "fighting all the time This movie is slower than soap opera and suddenly Jake decides to become Rambo " \
           "and kill the zombie OK first of all when you re going to make film you must Decide if its thriller or " \
           "drama As drama the movie is watchable Parents are divorcing arguing like in real life And then we " \
           "have Jake with his closet which totally ruins all the film expected to see BOOGEYMAN similar movie " \
           "and instead watched drama with some meaningless thriller spots out of just for the well playing " \
           "parents descent dialogs As for the shots with Jake just ignore them "

X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(sen)

y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

filename = "fin_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))

instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]
instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
prediction = loaded_model.predict(instance)
print(prediction)
print("execution time = ", time.time() - start, " seconds")
