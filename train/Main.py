import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# download stopwords and punkt using nltk
import nltk
nltk.download("stopwords")
nltk.download("punkt")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

import pickle

df = pd.read_csv("data/spam.csv")

# Data Splitting
X = df.Teks
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Training
stopwords_indo = stopwords.words("indonesian") + list(punctuation)
pipeline = Pipeline([
    ("prep", TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords_indo)),
    ("algo", LogisticRegression(random_state=42))
])
model = RandomizedSearchCV(pipeline, 
                           param_distributions={
                               "algo__C" : [1.2, 2.3, 3.2],
                               "algo__fit_intercept": [False, True]
                           },
                           cv=3,
                           verbose=1,
                           n_iter=50,
                           n_jobs=-1,
                           random_state=42
                           )

model.fit(X_train, y_train)

print(f"Best Params : {model.best_params_}")
print(f"Best Score : {model.best_score_}\n")
print(f"Train Score : {model.score(X_train, y_train)}")
print(f"Test Score : {model.score(X_test, y_test)}")

# Save Model
with open("model/spam.pkl", "wb") as file:
    pickle.dump(model, file)