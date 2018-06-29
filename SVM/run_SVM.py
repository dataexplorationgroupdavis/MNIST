# python3
from mnist import MNIST
import numpy as np
import pandas as pd
from sklearn import svm, metrics

mndata = MNIST('../samples')
Xtrain, ytrain = mndata.load_training()
Xtest, ytest = mndata.load_testing()

XtrainArr = np.array(Xtrain)
ytrainArr = np.array(ytrain)
XtestArr = np.array(Xtest)
ytestArr = np.array(ytest)

from time import time

result = {}
# we choose to use radial kernel
for C in [1, 5, 10, 100, 200]:
    for gamma in [0.001, 0.01, 0.05, 0.1, 0.5]:
        print('\n', "-------------------------", '\n')
        starttime = time()
        
        print(("C: {}, gamma: {}").format(C, gamma))
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(XtrainArr/255.0, ytrainArr)
        predictions = clf.predict(XtestArr/255.0)
        
        correct = sum(predictions == ytestArr)
        
        acc = correct / float(len(ytest))
        score = clf.score(XtestArr/255.0, ytestArr)
        timeUsed = time() - starttime
        print(("Accuracy: {}").format(acc))
        print(("Score: {}").format(score))
        print(("Time spent: {}").format(timeUsed))
        print(metrics.confusion_matrix(ytestArr, predictions))
        result['C'] = C
        result['gamma'] = gamma
        result['acc'] = acc
        result['score'] = score
        result['time'] = timeUsed

df = pd.DataFrame(result)
print(df)
