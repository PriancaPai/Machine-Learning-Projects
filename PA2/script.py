import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def pdfCalculator(mean,cov,X):

    pdf = np.divide(
    np.exp(
        np.divide(
            np.dot(
                np.dot(np.transpose(np.subtract(X,mean)),
                        inv(cov)),
                np.subtract(X,mean)),
            -2) ),
        np.multiply(np.power(det(cov),0.5) ,
                    (44/7)) )


    return pdf

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix



    # IMPLEMENT THIS METHOD

    #sigma^2 = 1/n*(Summation1-N(Xi-Ui))^2 (Use np.cov)

    covmat = np.cov(np.transpose(X))

    #Append y to X
    Xy = np.concatenate((X, y), axis=1)

    #Calculate Mean per class

    #Group by class
    means=[]
    for x in sorted(np.unique(Xy[...,2])):
        means.append(np.mean((Xy[np.where(Xy[...,2]==x)])[:, [0, 1]],axis=0))

    means = np.transpose(means)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    means=[]
    covmats = []

    #Append y to X
    Xy = np.concatenate((X, y), axis=1)

    #Group by class
    for x in sorted(np.unique(Xy[...,2])):
        means.append(np.mean((Xy[np.where(Xy[...,2]==x)])[:, [0, 1]],axis=0))
        covmats.append(np.cov(np.transpose((Xy[np.where(Xy[...,2]==x)])[:, [0, 1]])))

    means = np.transpose(means)


    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    ypred = np.empty([Xtest.shape[0], 1])

    for sample in range(Xtest.shape[0]):
        predict = 0
        i = 0
        for index in range(means.shape[1]):
            p = pdfCalculator(means[:,index],covmat,Xtest[sample])
            if p > predict:
                predict = p
                i = index
        ypred[sample,0] = (i+1)

    correct = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            correct = correct + 1;
    acc = correct

    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    ypred = np.empty([Xtest.shape[0], 1])

    for sample in range(Xtest.shape[0]):
        predict = 0
        i = 0
        for index in range(means.shape[1]):
            p = pdfCalculator(means[:,index],covmats[index],Xtest[sample])
            if p > predict:
                predict = p
                i = index
        ypred[sample,0] = (i+1)

    correct = 0;
    for i in range(Xtest.shape[0]):
        if ypred[i] == ytest[i]:
            correct = correct + 1;
    acc = correct

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    """Solution Start"""

    #Formula to implement
    # w = (inverse(transpose(X) * X)) * transpose(X) * y;

    # initialize w with zeroes
    w = np.zeros((X.shape[1], 1))

    # transpose(X) * X
    prod_tranposeX_X = np.dot(X.T, X)

    # (inverse(transpose(X) * X))
    prod_InverseOfTranposeX_X = inv(prod_tranposeX_X)

    # transpose(X) * y
    prod_TransposeX_Y = np.dot(X.T, y)

    # (inverse(transpose(X) * X)) * transpose(X) * y;
    w = np.dot(prod_InverseOfTranposeX_X, prod_TransposeX_Y)

    """Solution Ends"""
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    """Solution Start"""

    #Formula to implement
    # w =  inverse((λI + transpose(X) * X)) * transpose(X) * y;

    N = X.shape[0]
    d = X.shape[1]

    # I
    identityMat = np.identity(d)

    # transpose(X) * X)
    prod_TransposeX_X = np.dot(X.T, X)

    # λI
    prod_Lambd_Identity = lambd * identityMat

    # (λI + transpose(X) * X)
    sum_Prod_TransposeX_X_Prod_Lambd_Identity = prod_TransposeX_X + prod_Lambd_Identity

    # inverse((λI + transpose(X) * X))
    inverse_Sum_Prod_TransposeX_X_Prod_Lambd_Identity = inv(sum_Prod_TransposeX_X_Prod_Lambd_Identity)

    # transpose(X) * y
    prod_TransposeX_Y = np.dot(X.T, y)

    # inverse((λI + transpose(X) * X)) * transpose(X) * y
    w = np.dot(inverse_Sum_Prod_TransposeX_X_Prod_Lambd_Identity, prod_TransposeX_Y)

    """Solution Ends"""
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD

    """Solution Start"""

    #Formula to implement

    # mse = (1/N)(Summation(i:1,N)(Yi - transpose(w)*(Xi))^2)
    # rmse = (1/N)sqrt(mse)

    #N
    N = Xtest.shape[0]

    # transpose(w)*(Xi)
    weight_Xtest_Prod = np.dot(Xtest,w)

    # (Yi - transpose(w)*(Xi))
    y_Subtract_weight_Xtest = np.subtract(ytest, weight_Xtest_Prod)

    #(Yi - transpose(w)*(Xi))^2
    prod_Y_Subtract_weight_Xtest = np.multiply(y_Subtract_weight_Xtest, y_Subtract_weight_Xtest)

    # Summation(i:1,N)(Yi - transpose(w)*(Xi))^2
    summation_Prod_Y_Subtract_weight_Xtest = np.sum(prod_Y_Subtract_weight_Xtest)

    """NOTE"""
    summation_Prod_Y_Subtract_weight_Xtest = np.sum(prod_Y_Subtract_weight_Xtest)/N

    """Solution Ends"""
    mse = summation_Prod_Y_Subtract_weight_Xtest

    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    """Solution Start"""

    # Derive from Page 6 of the Handout

    #Formula to implement
    # J(w) =  (1/2)(Summation(i:1,N)(Yi - transpose(w)*(Xi))^2) + (1/2)λ(w*transpose(w));

    N = X.shape[0]

    error = 0

    # (Yi - transpose(w)*(Xi))
    y_Subtract_weight_Xtest = np.subtract(y.T, np.dot(w.T, X.T))

    # (Yi - transpose(w)*(Xi))^2)
    square_Y_Subtract_weight_Xtest = np.multiply(y_Subtract_weight_Xtest, y_Subtract_weight_Xtest)

    # (Summation(i:1,N)(Yi - transpose(w)*(Xi))^2)
    summation_Square_Y_Subtract_weight_Xtest = np.sum(square_Y_Subtract_weight_Xtest)

    # (w*transpose(w))
    prod_TransposeW_W = np.dot(w.T, w)

    # λ(w*transpose(w))
    prod_lamda_prod_TransposeW_W = np.multiply(lambd, prod_TransposeW_W)

    #(1/2)(Summation(i:1,N)(Yi - transpose(w)*(Xi))^2) + (1/2)λ(w*transpose(w));
    error = (summation_Square_Y_Subtract_weight_Xtest / 2) + (prod_lamda_prod_TransposeW_W / 2)


    #Formula to implement
    # gradient = transpose(X)(Xw−y) + λw
    #          = transpose(X)Xw - transpose(X)y + λw

    # λw
    prod_Lambda_W = np.multiply(lambd, w)

    # transpose(X)y
    prod_TansposeY_X = np.dot(y.T, X)

    # transpose(X)X
    prod_TransposeX_X = np.dot(X.T, X)

    # transpose(X)Xw
    prod_TransposeW_prod_TransposeX_X =  np.dot(w.T, prod_TransposeX_X)

    # λw - transpose(X)y
    subtract_Prod_Lambda_W_prod_TansposeY_X = np.subtract(prod_Lambda_W, prod_TansposeY_X)

    # transpose(X)Xw - transpose(X)y + λw
    error_grad = np.add(subtract_Prod_Lambda_W_prod_TansposeY_X, prod_TransposeW_prod_TransposeX_X)


    error = error.flatten()
    error_grad = error_grad.flatten()

    """Solution Ends"""
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))

    # IMPLEMENT THIS METHOD
    # 5 Handling Non-linear Relationship : Page 9 of Handouts

    N = x.shape[0]
    Xd = np.ones((N, p + 1));
    for i in range(1, p + 1):
        Xd[:, i] = x ** i;
    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

w = learnOLERegression(X,y)
mle = testOLERegression(w,X,y)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,X_i,y)

print('train data: MSE without intercept '+str(mle))
print('train data: MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))

optimal = 4000
l = 0

for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    if(mses3[i] < optimal):
        optimal = mses3[i]
        l = lambd
    i = i + 1

print('Optimal lambda '+str(l))
print('Optimal error '+str(optimal))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))

optimal_p_0 = 0
optimal_err_0 = 10000
optimal_p_1 = 0
optimal_err_1 = 10000

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

    if(mses5[p,0] < optimal_err_0):
        optimal_err_0 = mses5[p,0]
        optimal_p_0 = p

    if(mses5[p,1] < optimal_err_1):
        optimal_err_1 = mses5[p,1]
        optimal_p_1 = p


print('Optimal p - No regularization '+ str(optimal_p_0))
print('Optimal p  - With regularization '+ str(optimal_p_1))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
