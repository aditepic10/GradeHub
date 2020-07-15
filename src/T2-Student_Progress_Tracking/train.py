import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os

dir_path = os.path.dirname(os.path.realpath(__file__)).replace("src", "data")
df = pd.read_csv(dir_path + "/student-mat.csv")  # Read in data file as csv into a pandas data frame
t_start = time.perf_counter()
print(datetime.now())
# Preliminary transformation of data
enc = LabelEncoder()
for i in df.select_dtypes('object').columns:
    df[i] = enc.fit_transform(df[i])

X = df.drop(['school', 'G1', 'G2'], axis=1)  # Remove all grades except for final
y = df['G3']
sorted_y = sorted(y[i] for i in range(len(y)))
for i in range(len(y)):
    y.loc[i] = 1 if (sorted_y.index(y[i]) / len(sorted_y)) < 0.25 else 0

used_data = {"failures", "Medu", "higher", "age", "Fedu", "goout", "romantic"}  # Labels to be used
X = X.drop([label for label in X if label in used_data], axis=1)  # remove all irrelevant labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)  # split train/test data
scaler = StandardScaler()  # scale to transform data for neural network
scaler.fit(X_train)
X_train = scaler.fit_transform(X_train)  # transform data to be used in neural network
X_test = scaler.transform(X_test)  # transform data to be used in neural network

# Note: Talk about the scaling in the paper!
architecture = (5, 10)  # replace with any architecture
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=architecture, random_state=1)
# model = DecisionTreeClassifier()  # Decision tree model

print("Preprocessing time: %ss" % (round(time.perf_counter() - t_start, 3)))  # time to preprocess
print("Network architecture: %s" % list(architecture))  # architecture dims
t1 = time.perf_counter()
model.fit(X_train, y_train)  # training multi-layer perceptron
print("Training time: %ss" % (round(time.perf_counter() - t1, 3)))  # training time

y_predict, y_test = model.predict(X_test), list(y_test)  # predicting student grade using MLP

acc = sum(y_predict[n] == y_test[n] for n in range(len(y_predict))) / len(y_test)
false_at_risk = sum(y_predict[n] == 1 != y_test[n] for n in range(len(y_predict))) / len(y_predict) / (1 - acc)

print("Accuracy on test data (1/3 of original randomly picked): %s" % acc)  # comparing accuracy
print("Falsely identified at-risk students (pct of all misidentified values): %s" % false_at_risk)
print("Falsely ignored at-risk students (pct of all misidentified values): %s" % (1 - false_at_risk))
