print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import pickle
from numpy import asarray
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# The digits dataset
#digits = datasets.load_digits()
#print(len(digits.images))
#print(len(digits.target))
with open('DIGITS.txt', "rb") as fp:
    images = asarray(pickle.load(fp)).astype(np.float64)
with open('DIGIT_TARGETS.txt', "rb") as fp:
    target = [int(i) for i in pickle.load(fp)]
print(len(images))
print(len(target))
print(float(len([i for i in target if i == 0]))/float(len(target)))
print(float(len([i for i in target if i == 1]))/float(len(target)))
print(float(len([i for i in target if i == 2]))/float(len(target)))
print(float(len([i for i in target if i == 3]))/float(len(target)))
print(float(len([i for i in target if i == 4]))/float(len(target)))
print(float(len([i for i in target if i == 5]))/float(len(target)))
print(float(len([i for i in target if i == 6]))/float(len(target)))
print(float(len([i for i in target if i == 7]))/float(len(target)))
print(float(len([i for i in target if i == 8]))/float(len(target)))
print(float(len([i for i in target if i == 9]))/float(len(target)))
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(images, target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(images)
data = images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.5, shuffle=False)

# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

images_and_predictions = list(zip(images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()