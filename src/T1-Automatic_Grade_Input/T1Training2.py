# Standard scientific Python imports
import matplotlib.pyplot as plt
import pickle
from numpy import asarray
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
# The digits dataset
#digits = datasets.load_digits()
#print(len(digits.images))
#print(len(digits.target))
with open('digits.txt', "rb") as fp:
    images = asarray(pickle.load(fp)).astype(np.float64)
with open('DIGIT_TARGETS.txt', "rb") as fp:
    target = asarray([int(i) for i in pickle.load(fp)]).astype(np.float64)
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
_, axes = plt.subplots(2, 15)
images_and_labels = list(zip(images, target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:15]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('T:%i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images)
data = images.reshape((n_samples, -1))

architecture = (2750,)
classifier = MLPClassifier(solver='adam', learning_rate="adaptive", hidden_layer_sizes=architecture, random_state=1, verbose=True, alpha=1e-5, max_iter = 1000)
# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.1, shuffle = True)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

images_and_predictions = list(zip(images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:15]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('P:%i' % prediction)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

print(datetime.now())
print("Network architecture: " + str(architecture))
print("Percent of dataset used: " + str((float(len(images))/float(42000))*100.0))
print("Accuracy: " + str(accuracy_score(predicted, y_test)))

plt.show()