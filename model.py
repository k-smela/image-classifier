import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# skimage imports
from skimage.io import imread
from skimage.filters import sobel, sato, gaussian
from skimage.feature import hog
from skimage.transform import resize

# sklearn inputs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

size = 64

X = np.load('X_array.npy')
Y = np.load('Y_array.npy')

# main model class
# We wrote a Model class to be able to keep our code succinct and more readable
# In theory, this was supposed to be implemented in the GUI, but now
# this is really helpful to run specific instances of a model on the data set.
# This makes the code more efficient because all of the model we have
# implemented are not run every time that we run the file.
class Model():

    def __init__(self, classifier, X_train, X_test, Y_train, Y_test):
        # each of these variables come from the split function
        # the classifier depends on the model called
        self.classifier = classifier # which sklearn classification model used
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    # evaluate function will take in the data, fit the model, predict using X_test
    # and return the accuracy score associated with the instance of the model
    def evaluate(self):
        self.classifier.fit(self.X_train, self.Y_train)
        preds = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, preds)
        return accuracy

    def getpreds(self):
        self.classifier.fit(self.X_train, self.Y_train)
        preds = self.classifier.predict(self.X_test)
        return preds

# classifier subclasses
# each specific model uses the superclass's init and evaluate function,
# but now uses a specific classifier
class Log(Model):
# logistic regression for classification model
    def __init__(self, X_train, X_test, Y_train, Y_test):
        classifier = LogisticRegression(max_iter = 10000)
        super().__init__(classifier, X_train, X_test, Y_train, Y_test)

    def evaluate(self):
        return super().evaluate()

class MLP(Model):
#neural network model
    def __init__(self, X_train, X_test, Y_train, Y_test):
        classifier = MLPClassifier(max_iter = 10000)
        super().__init__(classifier, X_train, X_test, Y_train, Y_test)

    def evaluate(self):
        return super().evaluate()

class KNN(Model):
# K neareset neighbors model
    def __init__(self, X_train, X_test, Y_train, Y_test):
        classifier = KNeighborsClassifier()
        super().__init__(classifier, X_train, X_test, Y_train, Y_test)

    def evaluate(self):
        return super().evaluate()

class Ridge(Model):
# ridge classification model
    def __init__(self, X_train, X_test, Y_train, Y_test):
        classifier = RidgeClassifier()
        super().__init__(classifier, X_train, X_test, Y_train, Y_test)

    def evaluate(self):
        return super().evaluate()

class NaiveBayes(Model):
# Naive Bayes model
    def __init__(self, X_train, X_test, Y_train, Y_test):
        classifier = GaussianNB()
        super().__init__(classifier, X_train, X_test, Y_train, Y_test)

    def evaluate(self):
        return super().evaluate()

class Radius(Model):
# Radius Neighbors classification model
    def __init__(self, X_train, X_test, Y_train, Y_test):
        classifier = RadiusNeighborsClassifier(radius = 70)
        super().__init__(classifier, X_train, X_test, Y_train, Y_test)

    def evaluate(self):
        return super().evaluate()

# splitting up our data into testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# function that extracts necessary features and runs the fitted model on the user provided image
def single_image(coral_image): # takes in an image file name
    image = imread(coral_image)
    image_resized = resize(image, (size, size), mode = "reflect")

    # extracting color channel means
    red_channel = image_resized[:,:,0]
    green_channel = image_resized[:,:,1]
    blue_channel = image_resized[:,:,2]

    # averages of the color channels
    single_red_avg = np.mean(red_channel)
    single_green_avg = np.mean(green_channel)
    single_blue_avg = np.mean(blue_channel)

    # applying skimage filters and extracting features from the copy of the image
    single_gaus_avg = np.mean(gaussian(image_resized))
    single_gaus_max = np.max(gaussian(image_resized))
    single_gaus_min = np.min(gaussian(image_resized))
    single_gaus_std = np.std(gaussian(image_resized))

    single_sob_avg = np.mean(sobel(image_resized))
    single_sob_max = np.max(sobel(image_resized))
    single_sob_min = np.min(sobel(image_resized))
    single_sob_std = np.std(sobel(image_resized))

    single_sato_avg = np.mean(sato(image_resized))
    single_sato_max = np.max(sato(image_resized))
    single_sato_min = np.min(sato(image_resized))
    single_sato_std = np.std(sato(image_resized))

    im_left_avg = np.mean(image_resized[:,:size//2, :])
    im_right_avg = np.mean(image_resized[:,size//2:, :])
    im_max = np.max(image_resized)
    im_min = np.min(image_resized)
    im_std = np.std(image_resized)

    # creating vector of the image's extracted features
    # this is our vector (list)that consists of all of the collected
    # features of the single image that is passed through this function.
    single_image_feature_vector = [float(single_red_avg), float(single_green_avg),
                                    float(single_blue_avg), float(single_gaus_avg),
                                    float(single_sob_avg), float(single_sato_avg),
                                    float(single_gaus_max), float(single_sob_max),
                                    float(single_sato_max),
                                    float(single_gaus_min), float(single_sob_min),
                                    float(single_sato_min),
                                    float(single_gaus_std), float(single_sob_std),
                                    float(single_sato_std),
                                    float(im_left_avg), float(im_right_avg),
                                    float(im_max), float(im_min), float(im_std)]

    # turning feature vector into 2d np array so that we can pass it through
    # the classification model
    single_array = np.array(single_image_feature_vector).reshape(-1,len(single_image_feature_vector))

    # applying model to user image
    model = KNN(X_train, X_test, Y_train, Y_test)
    model.classifier.fit(X_train, Y_train)
    prediction = model.classifier.predict(single_array)
    print(prediction[0])

# running the function with the user file name
single_image("coral.jpg")

#Lists to store important variables for the matplotlib charts
models = [Log , MLP, KNN, Ridge, NaiveBayes, Radius]
#Each item is an instance of the corresponing models class
modelnames = ["LogisticRegression", "MLPClassifier", "KNeighborsClassifier", "RidgeClassifier", "NaiveBayes","RadiusNeighborsClassifier"]
accuracy = []#stores the model accuracies to be displayed
cm = []#Stores confusion matrices for each model to be displayed
i=0
baseline = []#stores the baseline model values for each model
# providing accuracy scores for each of our models and looping to store items
#that are important in matplotlib charts

for model in models:
    try_model = model(X_train, X_test, Y_train, Y_test)
    score = try_model.evaluate()
    print(f'Accuracy for {model.__name__} model: {score}')
    baseline.append(score)
    accuracy.append(score)
    cm.append(confusion_matrix(Y_test, try_model.getpreds()))
    i += 1


#Code for printing each of the confusion matrices with labels for X and Y axises
for i in range(len(modelnames)):
    labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    labels2 = np.asarray(labels).reshape(2,2)
    sns.heatmap(data = cm[i], annot = labels2, fmt="")
    plt.suptitle(modelnames[i] +  " confusion matrix")
    plt.xlabel("POSITIVE                            NEGATIVE")
    plt.ylabel("POSITIVE                            NEGATIVE")
    plt.show()

#dictionary used to store the values from the accuracy loop on lines 169-175
#and the corresponding Model Names, then displaying those values in a barplot
accuracydict = {"Accuracy":accuracy, "Model":["LogisticRegression", "MLPClassifier", "KNeighborsClassifier", "RidgeClassifier", "NaiveBayes", "RadiusNeighborsClassifier"]}
df = pd.DataFrame(accuracydict)
sns.barplot(data=df, x="Model", y="Accuracy", palette = "pastel")
plt.suptitle("Accuracy of all the models")
plt.show()



####The following code involves all the computing for getting the impact on the
#accuracy when a certain feature is removed. The code takes in the array of X
# and then makes a copy, uses that copy to apply the model, however before trianing
#Feature N is removed. The accuracy is recorded and then subtracted by the baseline
#for each model.
featurenames= ["single red avg", "single green avg","single blue avg", "single gaus avg",
                "single sobel avg", "single sato avg",
                "single gaus max", "single sob max", "single sato max",
                "single gaus min", "single sob min", "single sato min",
                "single gaus std", "single sob std", "single sato std",
                "im left avg", "im right avg", "im max", "im min", "im std"]

#feature accuracy for LogisticRegression
feature_accuracieslogreg =[]

for N in range(X.shape[1]):
    Xcopy = X.copy()
    Xcopy[:,N] = 0
    tX_train, tX_test, tY_train, tY_test = train_test_split(Xcopy, Y, test_size = 0.2, random_state = 42)
    tempclassifier = LogisticRegression(max_iter = 10000)
    tempclassifier.fit(tX_train, tY_train)
    temppredictions = tempclassifier.predict(tX_test)
    tempaccuracy = accuracy_score(temppredictions, tY_test)-baseline[0]
    feature_accuracieslogreg.append(tempaccuracy)

feataccuracydictlogreg = {"Feature Accuracies":feature_accuracieslogreg, "Removed Feature":featurenames}
dfLogReg = pd.DataFrame(feataccuracydictlogreg)
#print(dfLogReg)
sns.barplot(data = dfLogReg, x= "Feature Accuracies", y ="Removed Feature",palette = "pastel")
plt.suptitle("Accuracies for LogisticRegression after removing Feature")
plt.show()


##feature accuracy for MLPClassifier
feature_accuraciesMLP = []
for N in range(X.shape[1]):
    Xcopy = X.copy()
    Xcopy[:,N] = 0
    tX_train, tX_test, tY_train, tY_test = train_test_split(Xcopy, Y, test_size = 0.2, random_state = 42)
    tempclassifier = MLPClassifier(max_iter = 10000)
    tempclassifier.fit(tX_train, tY_train)
    temppredictions = tempclassifier.predict(tX_test)
    tempaccuracy = accuracy_score(temppredictions, tY_test)-baseline[1]
    feature_accuraciesMLP.append(tempaccuracy)

feataccuracydictMLP = {"Feature Accuracies":feature_accuraciesMLP, "Removed Feature":featurenames}
dfMLP = pd.DataFrame(feataccuracydictMLP)
#print(dfMLP)
sns.barplot(data = dfMLP, x= "Feature Accuracies", y ="Removed Feature",palette = "pastel")
plt.suptitle("Accuracies for MLPClassifier after removing Feature")
plt.show()

#feature accuracy for KNeighborsClassifier
feature_accuraciesKNN = []
for N in range(X.shape[1]):
    Xcopy = X.copy()
    Xcopy[:,N] = 0
    tX_train, tX_test, tY_train, tY_test = train_test_split(Xcopy, Y, test_size = 0.2, random_state = 42)
    tempclassifier = KNeighborsClassifier()
    tempclassifier.fit(tX_train, tY_train)
    temppredictions = tempclassifier.predict(tX_test)
    tempaccuracy = accuracy_score(temppredictions, tY_test)-baseline[2]
    feature_accuraciesKNN.append(tempaccuracy)

feataccuracydictKNN = {"Feature Accuracies":feature_accuraciesKNN, "Removed Feature":featurenames}
dfKNN = pd.DataFrame(feataccuracydictKNN)
#print(dfKNN)
sns.barplot(data = dfKNN, x= "Feature Accuracies", y ="Removed Feature",palette = "pastel")
plt.suptitle("Accuracies for KNeighborsClassifier after removed Feature")
plt.show()

#feature accuracy for RidgeClassifier
feature_accuraciesR = []
for N in range(X.shape[1]):
    Xcopy = X.copy()
    Xcopy[:,N] = 0
    tX_train, tX_test, tY_train, tY_test = train_test_split(Xcopy, Y, test_size = 0.2, random_state = 42)
    tempclassifier = RidgeClassifier()
    tempclassifier.fit(tX_train, tY_train)
    temppredictions = tempclassifier.predict(tX_test)
    tempaccuracy = accuracy_score(temppredictions, tY_test)-baseline[3]
    feature_accuraciesR.append(tempaccuracy)

feataccuracydictR = {"Feature Accuracies":feature_accuraciesR, "Removed Feature":featurenames}
dfR = pd.DataFrame(feataccuracydictR)
#print(dfR)
sns.barplot(data = dfR, x= "Feature Accuracies", y ="Removed Feature",palette = "pastel")
plt.suptitle("Accuracies for RidgeClassifier after removed Feature")
plt.show()

#feature accuracy for NaiveBayes
feature_accuraciesNB = []
for N in range(X.shape[1]):
    Xcopy = X.copy()
    Xcopy[:,N] = 0
    tX_train, tX_test, tY_train, tY_test = train_test_split(Xcopy, Y, test_size = 0.2, random_state = 42)
    tempclassifier = GaussianNB()
    tempclassifier.fit(tX_train, tY_train)
    temppredictions = tempclassifier.predict(tX_test)
    tempaccuracy = accuracy_score(temppredictions, tY_test)-baseline[4]
    feature_accuraciesNB.append(tempaccuracy)

feataccuracydictNB = {"Feature Accuracies":feature_accuraciesNB, "Removed Feature":featurenames}
dfNB = pd.DataFrame(feataccuracydictNB)
#print(dfNB)
sns.barplot(data = dfNB, x= "Feature Accuracies", y ="Removed Feature",palette = "pastel")
plt.suptitle("Accuracies for NaiveBayes after removed Feature")
plt.show()


#feature accuracy for RadiusNeighborsClassifier
feature_accuraciesRC = []
for N in range(X.shape[1]):
    Xcopy = X.copy()
    Xcopy[:,N] = 0
    tX_train, tX_test, tY_train, tY_test = train_test_split(Xcopy, Y, test_size = 0.2, random_state = 42)
    tempclassifier = RadiusNeighborsClassifier()
    tempclassifier.fit(tX_train, tY_train)
    temppredictions = tempclassifier.predict(tX_test)
    tempaccuracy = accuracy_score(temppredictions, tY_test)-baseline[5]
    feature_accuraciesRC.append(tempaccuracy)

feataccuracydictRC = {"Feature Accuracies":feature_accuraciesRC, "Removed Feature":featurenames}
dfRC = pd.DataFrame(feataccuracydictRC)
#print(dfRC)
sns.barplot(data = dfRC, x= "Feature Accuracies", y ="Removed Feature",palette = "pastel")
plt.suptitle("Accuracies for RadiusNeighborsClassifier after removed Feature")
plt.show()
