import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

df = pd.read_csv("student-mat.csv")  # Read in data file as csv into a pandas data frame
df.head()
t_start = time.perf_counter()
print(datetime.now())
# Preliminary transformation of data
enc = LabelEncoder()
category_colums = df.select_dtypes('object').columns
for i in category_colums:
    df[i] = enc.fit_transform(df[i])

df.head()

X = df.drop(['school', 'G1', 'G2'], axis=1)  # Remove all grades except for final
y = df['G3']
sorted_y = sorted(y[i] for i in range(len(y)))
for i in range(len(y)):
    orig = y[i]  # save original value
    # at-risk defined as percentile < 0.10 (tent.)
    y.loc[i] = 1 if (sorted_y.index(y[i]) / len(sorted_y)) < 0.10 else 0
    sorted_y.remove(orig)  # remove original value from sorted list

used_data = {"failures", "Medu", "higher", "age", "Fedu", "goout", "romantic"}  # Labels to be used
X = X.drop([label for label in X if label not in used_data], axis=1)  # remove all irrelevant labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # split train/test data
sc = StandardScaler()  # scale to transform data for neural network
X_train = sc.fit_transform(X_train)  # transform data to be used in neural network
X_test = sc.transform(X_test)  # transform data to be used in neural network

architecture = (1000000000, 5)
model = MLPClassifier(solver='lbfgs', learning_rate="adaptive", hidden_layer_sizes=architecture, random_state=1)
# model = DecisionTreeClassifier()  # Decision tree model

print("Preprocessing time: %ss" % (round(time.perf_counter() - t_start, 3)))
print("Network architecture: %s" % list(architecture))
t1 = time.perf_counter()
model.fit(X_train, y_train)
print("Training time: %ss" % (round(time.perf_counter() - t1, 3)))
y_predict = model.predict(X_test)
print("Accuracy on test data (1/3 of original randomly picked): %s" % accuracy_score(y_predict, y_test))
