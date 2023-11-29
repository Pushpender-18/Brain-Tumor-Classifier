import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

s, e = 3600, 3606
a = {1:"Present", 0: "Absent"}

data = pd.read_csv('Brain Tumor.csv')

labels = data['Class'].to_numpy()
labels = labels[s: e]
testData = data.drop(['Image', 'Class'], axis=1).to_numpy()
testData = testData[s: e]
image = []

for i in range(s, e):
    x = Image.open(f'Brain Tumor/Image{i}.jpg')
    image.append(np.asarray(x))


mx = MinMaxScaler()

mx.fit_transform(testData)

testData = mx.transform(testData)

testData = pd.DataFrame(testData)

dataClf = pickle.load(open('nimg.md', 'rb'))

dataClfPrediction = dataClf.predict(testData)

dataClfAcc = metrics.accuracy_score(labels, dataClfPrediction)

print(dataClfAcc)

for i in range(len(dataClfPrediction)):
    plt.imshow(image[i])
    plt.title(f"Actual : Brain Tumor {a[labels[i]]}\nPredicted : Brain Tumor {a[dataClfPrediction[i]]}")
    plt.show()