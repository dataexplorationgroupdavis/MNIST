# Try support vector machine algorithm in R.
# methods() # checking possible methods of a generic function
# getAnywhere() # checking source code for a method


# simulate data
set.seed(1)
x = matrix(rnorm(20*2), ncol=2)
y = c(rep(-1, 10), rep(1, 10))

x[y==1, ] = x[y==1, ] + 1
# plot(x, col=(3-y))

dat = data.frame(x=x, y=as.factor(y))
library(e1071)
svmfit = svm(y~., data=dat, kernel='linear', cost=10, scale=FALSE)
plot(svmfit, dat)

# tuning parameter C
tune.out = tune(svm, y~., data=dat, kernel='linear',ranges=list(cost=c(0.1,1,5,10,100)))
print('tunning parameter')
print(tune.out)

bestmod = tune.out$best.model
print('best model summary')
print(summary(bestmod))

xtest = matrix(rnorm(20*2), ncol=2)
ytest = sample(c(-1,1), 20, rep=TRUE)

xtest[ytest==1,] = xtest[ytest==1,] + 1
testdat = data.frame(x=xtest, y=as.factor(ytest))

ypred = predict(bestmod, testdat)
print('prediction table')
print(table(predict=ypred, truth=testdat$y))

# with larger dataset
x = matrix(rnorm(200*2), ncol=2)
x[1:100, ] = x[1:100, ]+2
x[101:150, ] = x[101:150,] -2
y = c(rep(1,150),rep(2, 50))
dat = data.frame(x=x, y=as.factor(y))

plot(x, col=y) # the class of y is factor
train = sample(200, 100)
svmfit = svm(y~., data=dat[train, ], kernel='radial', gamma=1, cost=1)
plot(svmfit, dat[train,])

# incomplete
# ROC curve
# library(ROCR)
# rocplot = function(pred, truth, ...){
#    predob = prediction(pred, truth)
#    perf = performance(prodob, 'tpr', 'fpr')
#    plot(perf, ...)
# }
#
# svmfit.opt = svm(y~., data=dat[train,], kernel='radial', gamma=2, cost=1, decision.values=T)
# fitted = attributes(predict(svmfit.opt, dat[train,

