import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os

dir_path = os.path.dirname(os.path.realpath(__file__)).replace("src", "data")
df = pd.read_csv(dir_path + "/student-mat.csv")  # Read in data file as csv into a pandas data frame
t_start = time.perf_counter()
print(datetime.now())
# Preliminary transformation of data
enc = LabelEncoder()
category_colums = df.select_dtypes('object').columns
for i in category_colums:
    df[i] = enc.fit_transform(df[i])

X = df.drop(['school', 'G1', 'G2'], axis=1)  # Remove all grades except for final
y = pd.DataFrame(df["G3"], columns=["G3"])  # first = at_risk? second = not_at_risk
sorted_y = sorted(y["G3"][i] for i in range(len(y)))
y2_helper = []
for i in range(len(y)):
    orig = y["G3"][i]  # save original value
    # at-risk defined as percentile < 0.10 (tent.)
    y["G3"].loc[i] = 1 if (sorted_y.index(y["G3"][i]) / len(sorted_y)) < 0.10 else 0  # 1 = at_risk
    y2_helper.append(0 if (y["G3"].loc[i] == 1) else 1)  # binary values; opposite of at_risk
    sorted_y.remove(orig)  # remove original value from sorted list

used_data = {"failures", "Medu", "higher", "age", "Fedu", "goout", "romantic"}  # Labels to be used
X = X.drop([label for label in X if label not in used_data], axis=1)  # remove all irrelevant labels
y["G3-2"] = y2_helper  # add new label; complement of first
# print(y.head)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # split train/test data
sc = StandardScaler()  # scale to transform data for neural network
X_train = sc.fit_transform(X_train)  # transform data to be used in neural network
X_test = sc.transform(X_test)  # transform data to be used in neural network

architecture = (198,)  # replace with any architecture
model = MLPClassifier(solver='lbfgs', learning_rate="adaptive", hidden_layer_sizes=architecture, random_state=1,
                      max_iter=100000000)
# model = DecisionTreeClassifier()  # Decision tree model

print("Preprocessing time: %ss" % (round(time.perf_counter() - t_start, 3)))  # time to preprocess
print("Network architecture: %s" % list(architecture))  # architecture dims
t1 = time.perf_counter()
model.fit(X_train, y_train)  # training multi-layer perceptron
print("Training time: %ss" % (round(time.perf_counter() - t1, 3)))  # training time
y_predict = model.predict(X_test)  # predicting student grade using MLP

y_test = list(y_test)
false_at_risk = sum(y_predict[n] != y_test[n] for n in range(len(y_predict)) if y_predict[n] == 1) / len(y_test)
false_fine = sum(y_predict[n] != y_test[n] for n in range(len(y_predict)) if y_predict[n] == 0) / len(y_test)
acc = sum(y_predict[n] == y_test[n] for n in range(len(y_predict))) / len(y_test)
false_at_risk = false_at_risk / (1 - acc)
false_fine = false_fine / (1 - acc)
print("Accuracy on test data (1/3 of original randomly picked): %s" % acc)  # comparing accuracy
print("Falsely identified at-risk students (pct of all misidentified values): %s" % false_at_risk)
print("Falsely ignored at-risk students (pct of all misidentified values): %s" % false_fine)
