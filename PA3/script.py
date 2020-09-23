import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    w = initialWeights
    #print("w",w.shape)

    x = train_data
    #print("x",x.shape)

    y = labeli
    #print("y",y.shape)

    x = np.insert(x, 0, 1, axis=1)
    #print("x",x.shape)

    ##########################################################################################
    # ERROR Calculation
    ##########################################################################################

    theta = sigmoid(np.dot(x,w))
    #print("theta",theta.shape)

    theta = np.reshape(theta,(n_data,1))
    #print("theta",theta.shape)

    ln_theta = np.transpose(np.log(theta))
    #print("ln_theta",ln_theta.shape)

    e_1_minus_y = np.subtract(1,y)
    #print("e_1_minus_y",e_1_minus_y.shape)

    e_1_minus_theta = np.subtract(1,theta)
    #print("e_1_minus_theta",e_1_minus_theta.shape)

    e_ln_1_minus_theta = np.log(e_1_minus_theta)
    #print("e_ln_1_minus_theta",e_ln_1_minus_theta.shape)

    part_b = np.dot(e_1_minus_y.T,e_ln_1_minus_theta)
    #print("part_b",part_b.shape)

    part_a = np.dot(ln_theta,y)
    #print("part_a",part_a.shape)

    t_err = part_a+part_b
    #print("t_err",t_err.shape)

    error = (-1) * (t_err/ n_data)
    #print("error",error.shape)

    ##########################################################################################
    # ERROR GRAD Calculation
    ##########################################################################################


    eg_theta_minus_y = np.subtract(theta,y)
    #print("eg_theta_minus_y",eg_theta_minus_y.shape)

    t_eg = x*eg_theta_minus_y
    #print("t_eg",t_eg.shape)

    t_eg = np.sum(t_eg,axis=0)
    #print("t_eg",t_eg.shape)

    t_eg = np.reshape(t_eg,(n_features+1,1))
    #print("t_eg",t_eg.shape)

    eg = t_eg*(1/n_data)
    #print("eg",eg.shape)

    error_grad = eg
    #print("error_grad",error_grad.shape)

    ##########################################################################################
    #print("#########################")
    #print("error",error)
    #print("#########################")


    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    x = data
    x = np.insert(x, 0, 1, axis=1)
    w = W

    posterior = np.dot(x,w)
    pred = sigmoid(posterior)
    prediction = np.argmax(pred, axis=1)
    label = np.reshape(prediction,(data.shape[0],1))

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    error_grad.flatten()

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #Create bias term and add to data
    ones_bias = np.ones((n_data,1))

    train_data = np.concatenate((ones_bias,train_data),axis=1)

    #Create weight by reshaping
    W = params.reshape(n_feature + 1,n_class)

    #creating theta matrix
    theta = np.zeros((n_data, n_class))

    # For multi-class Logistic Regression, the posterior probabilities are given by a softmax transformation of
    # linear functions of the feature variables Equation(5)
    dot_product_W_X = np.dot(train_data, W)

    theta_matrix_sum = np.sum(np.exp(dot_product_W_X), axis=1).reshape(n_data, 1)

    theta_matrix = (np.exp(dot_product_W_X) / theta_matrix_sum)

    #log of theta matrix
    log_theta_matrix = np.log(theta_matrix)

    # The likelihood function with the negative logarithm Equation(7)
    error = (-1) * np.sum(np.sum(labeli * log_theta_matrix))

    # We now take the gradient of the error function with respect to one of the parameter vectors
    # w(k).  Making use of the result for the derivatives of the softmax function, we obtain: Equation(8)
    subtract_theta_y = theta_matrix - labeli

    error_grad = (np.dot(train_data.T, subtract_theta_y))

    #error = error / n_data
    #error_grad = error_grad / n_data

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    #Create bias term and add to data
    ones_bias = np.ones((data.shape[0],1))
    train_data = np.concatenate((ones_bias,data),axis=1)

    # For multi-class Logistic Regression, the posterior probabilities are given by a softmax transformation of
    # linear functions of the feature variables Equation(5)
    dot_product_W_X = np.dot(train_data, W)
    theta_matrix_sum = np.sum(np.exp(dot_product_W_X))

    posterior = np.exp(dot_product_W_X) / theta_matrix_sum

    for i in range(posterior.shape[0]):
        label[i] = np.argmax(posterior[i])
    label = label.reshape(label.shape[0], 1)

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

print("***********************************LINEAR KERNEL*********************************************")
# Linear kernel Implementation
start_time = time.time()
clf = SVC(kernel='linear')
clf.fit(train_data, train_label.flatten())
print('Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')
print("--- %s seconds ---" % (time.time() - start_time))


print("***********************************RADIAL KERNEL 0*********************************************")
# Radial kernel Implementation with Gamma value set to 1
start_time = time.time()
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label.flatten())
print('Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')
print("--- %s seconds ---" % (time.time() - start_time))


print("*********************************RADIAL KERNEL 1***********************************************")
# Radial kernel Implementation with Gamma value set to 0 i.e default
start_time = time.time()
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label.flatten())
print('Training set Accuracy:' + str(100*clf.score(train_data, train_label)) + '%')
print('Validation set Accuracy:' + str(100*clf.score(validation_data, validation_label)) + '%')
print('Testing set Accuracy:' + str(100*clf.score(test_data, test_label)) + '%')
print("--- %s seconds ---" % (time.time() - start_time))


print("***********************************RADIAL KERNEL C VALUES*********************************************")
# Radial kernel Implementation with multiple values of C i.e 1, 10, 20, 30, ..., 100
start_time = time.time()
training_accuracy = np.zeros(11)
validation_accuracy = np.zeros(11)
testing_accuracy = np.zeros(11)
cValues = np.zeros(11)
cValues[0] = 1.0
cValues[1:] = [x for x in np.arange(10.0, 101.0, 10.0)]
for i in range(11):
    clf = SVC(C=cValues[i],kernel='rbf')
    clf.fit(train_data, train_label.flatten())
    training_accuracy[i] = 100*clf.score(train_data, train_label)
    validation_accuracy[i] = 100*clf.score(validation_data, validation_label)
    testing_accuracy[i] = 100*clf.score(test_data, test_label)
    print('Training set Accuracy::: C Value = ' + str(cValues[i]) + str(":::") + str(training_accuracy[i]) + '%')
    print('Validation set Accuracy::: C Value = ' + str(cValues[i]) + str(":::") + str(validation_accuracy[i]) + '%')
    print('Testing set Accuracy::: C Value = ' + str(cValues[i]) + str(":::") +str(testing_accuracy[i]) + '%')

#Pickle File
#pickle.dump((training_accuracy, validation_accuracy, testing_accuracy),open("multipleCValuesRBF.pickle","wb"))

print("--- %s seconds ---" % (time.time() - start_time))

#Graph
plot(cValues, training_accuracy, 'o-',
    cValues, validation_accuracy,'o-',
    cValues, testing_accuracy, 'o-')


title('SVM with Gaussian kernel for multiple values of C')
legend(('Train','Validation','Test'), loc='upper left')
xlabel('C Values')
ylabel('Accuracy (%)')
grid(True)
savefig("multipleCValuesRBF.png")
show()

print("********************************************************************************")


"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
