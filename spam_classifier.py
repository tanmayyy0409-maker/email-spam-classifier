import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv")

cv = CountVectorizer()

X = cv.fit_transform(data["text"])

y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)

email = input("Enter Email Text: ")

msg = [email]

msg_count = cv.transform(msg)

prediction = model.predict(msg_count)

print("Prediction:", prediction[0])