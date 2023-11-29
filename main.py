from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle

images = []
 ## Changing directory
for i in range(1, 3763):
    x = np.asarray(Image.open(f"Brain Tumor/Image{i}.jpg").convert('L')) ## Loading image, converting to numpy array
    images.append(x) ## Appending image to list

data = np.array(images) ## Converting image list to numpy array
 
labels = pd.read_csv('Brain Tumor.csv') ## Loading csv file

labels = labels["Class"]

labels = labels.to_numpy()
data = data.reshape((len(labels), -1))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

sc = StandardScaler()

sc.fit_transform(x_train)

x_train, x_test = sc.transform(x_train), sc.transform(x_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

clf = LogisticRegression(max_iter=400) ## Logistic Regression classifier

##clf.fit(x_train, y_train) ## Training model
##predicted = clf.predict(x_test) ## Testing model

##os.chdir("/home/pushpender/Documents/BrainTumor/")
##pickle.dump(clf, open("lr_sc.md", 'wb')) ## Saving model

##print(metrics.accuracy_score(y_test, predicted)*100) ## Printing accurarcy of model

##metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted) ## Showing confusion matrix
##plt.show()

clf = pickle.load(open('lr_sc.md', 'rb'))
predicted = clf.predict(x_test)

print(metrics.accuracy_score(y_test, predicted)*100)