# python3
from mnist import MNIST
import numpy as np
from sklearn import svm

mndata = MNIST('samples')
Xtrain, ytrain = mndata.load_training()
Xtest, ytest = mndata.load_testing()

XtrainArr = np.array(Xtrain)
ytrainArr = np.array(ytrain)
XtestArr = np.array(Xtest)
ytestArr = np.array(ytest)

# take subsets to run svm
indexesList = [np.random.choice(range(60000), size=s, replace=False) for s in [100, 1000, 10000, 20000, 60000]]

from time import time

for indexes in indexesList:
    starttime = time()
    XtrainTemp = XtrainArr[indexes]
    ytrainTemp = ytrainArr[indexes]

    print(("training size: {}").format(len(indexes)))
    # print("start training")
    clf = svm.SVC(kernel='linear')
    clf.fit(XtrainTemp, ytrainTemp)
    
    # print("start testing")
    predictions = clf.predict(XtestArr)
    correct = sum(predictions == ytestArr)
    
    acc = correct / float(len(ytest))
    
    print(("Accuracy: {}").format(acc))
    print(("Time spent: {}").format(time() - starttime))
    print('\n', "-------------------------", '\n')
