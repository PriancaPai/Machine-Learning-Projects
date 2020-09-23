import numpy as np
import pickle
from time import time
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def one_K_coding_scheme(training_label,n_class):
    size = np.size(training_label)
    out = np.zeros((size, n_class), dtype=np.int)
    for i in range(size):
        index = int(training_label[i])
        out[i][index] = 1
    return out

def feed_Forward_Func(training_data, w1, w2):
    
    data = training_data.transpose() 

    data_bias = np.ones((1, np.size(data, 1)), dtype = np.int)

    data = np.concatenate((data, data_bias), axis = 0)       
    
    #Hidden Layer Start
    
    #Equation 1
    hidden_layer_intermediate = np.dot(w1, data)
    
    #Equation 2
    hidden_layer_output = sigmoid(hidden_layer_intermediate)  
    
    #Hidden Layer End
    
    #Output Layer Start
    
    hidden_layer_bias = np.ones((1,np.size(hidden_layer_output,1)),dtype = np.int)
    
    hidden_layer_output = np.concatenate((hidden_layer_output, hidden_layer_bias), axis = 0) 
    
    #Equation 3
    output_layer_intermediate = np.dot(w2, hidden_layer_output)                                            
    
    #Equation 4
    output_layer_output = sigmoid(output_layer_intermediate)
    
    #Output Layer End
    
    return data, hidden_layer_output, output_layer_output



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    """
    Following snippet is taken from the gradiance quiz
    """
    
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return  s

def preprocess_small():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_sample.mat'.

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

     - feature selection"""


    mat = loadmat('/home/anurag/Desktop/mnist_sample.mat')
            # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(4996, 784))
    validation_preprocess = np.zeros(shape=(1000, 784))
    test_preprocess = np.zeros(shape=(996, 784))
    train_label_preprocess = np.zeros(shape=(4996,))
    validation_label_preprocess = np.zeros(shape=(1000,))
    test_label_preprocess = np.zeros(shape=(996,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    counter = 0

    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            #print(counter,key)
            #counter = counter + 1
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 100  # defines the number of examples which will be added into the training set
            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[100:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 100] = tup[tup_perm[0:100], :]
            validation_len += 100

            validation_label_preprocess[validation_label_len:validation_label_len + 100] = label
            validation_label_len += 100

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    #print(train_len,validation_preprocess.shape,test_preprocess.shape)
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]
    
    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]
    
    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    feature_count = train_data.shape[1]
    
    print(feature_count)
    
    minList = np.amin(train_data, axis=0)
    maxList = np.amax(train_data, axis=0)
    
    redundant_feature_list = []
    
    active_feature_list = []

    for i in range(feature_count):
        if minList[i] == maxList[i]:
            redundant_feature_list.append(i)
        else:
            active_feature_list.append(i)

    train_data = np.delete(train_data, redundant_feature_list, axis=1)
    validation_data = np.delete(validation_data, redundant_feature_list, axis=1)
    test_data = np.delete(test_data, redundant_feature_list, axis=1)
    
    print('preprocess done')
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
def preprocess():
    """ Input:
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

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.

    feature_count = train_data.shape[1]
        
    minList = np.amin(train_data, axis=0)
    maxList = np.amax(train_data, axis=0)
    
    redundant_feature_list = []
    active_feature_list = []

    for i in range(feature_count):
        if minList[i] == maxList[i]:
            redundant_feature_list.append(i)
        else:
            active_feature_list.append(i)

    train_data = np.delete(train_data, redundant_feature_list, axis=1)
    validation_data = np.delete(validation_data, redundant_feature_list, axis=1)
    test_data = np.delete(test_data, redundant_feature_list, axis=1)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label, active_feature_list


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    obj_val = 0
    obj_grad = np.array([])
    
    data_count = training_data.shape[0]
    
    #1 of K Coding of labels Start
    labels = one_K_coding_scheme(training_label, n_class)
    #1 of K Coding of labels End
    
    labels = labels.transpose()
    
    """******** Feed Forward Start ********"""
    #Equation 1, 2, 3 and 4
    data, hidden_layer_output, output_layer_output = feed_Forward_Func(training_data, w1, w2)
    
    """******** Feed Forward End ********"""    
    
    """******** Bacpropagagtion Start ********"""
    #Equation 5
    log_likelihood_error_function = labels*np.log(output_layer_output) + (1 - labels)*np.log(1 - output_layer_output) 
    
    #Equation 6 and 7
    log_likelihood_error = (-1)*(np.sum(log_likelihood_error_function[:])/data_count)
        
    #Equation 8 and 9
    output_layer_delta = (output_layer_output - labels)
    
    w2_error = np.dot(output_layer_delta , hidden_layer_output.transpose())
    
    #Equation 10, 11 and 12
    hidden_layer_delta =  np.dot(w2.transpose(), output_layer_delta)*(hidden_layer_output*(1 - hidden_layer_output))
    
    w1_error = np.dot(hidden_layer_delta , data.transpose())
    w1_error = w1_error[:-1,:]
    
    #Equation 15
    regularization_term = ((np.sum(w1**2) + np.sum(w2**2))/(2*data_count))*lambdaval
 
    obj_val = log_likelihood_error + regularization_term
    
    #Equation 16 and 17
    w1_gradient = (w1_error + lambdaval*w1)/data_count
    w2_gradient = (w2_error + lambdaval*w2)/data_count
    
    """******** Bacpropagagtion End ********"""
    
    obj_grad = np.concatenate((w1_gradient.flatten(),w2_gradient.flatten()), axis = 0) 
    print("obj val",obj_val)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    training_data, hidden_layer_output, output_layer_output = feed_Forward_Func(data, w1, w2)
    labels = np.argmax(output_layer_output,axis=0)

    return labels


"""**************Neural Network Script Starts here********************************"""

#train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess_small()
train_data, train_label, validation_data, validation_label, test_data, test_label, feature_list = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]


# set the number of nodes in hidden unit (not including bias unit)
#n_hidden = 1

# set the number of nodes in output unit
n_class = 10

n_hidden = 50

print("**************************************************************************************")
# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)       
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

lambdaval = 10
print("lambdaVal : ",lambdaval)
print("Hidden : ",n_hidden)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. 
#Check documentation for a working example

opts = {'maxiter': 100}  # Preferred value.

T1 = time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
T2 = time()

print("Training Time : ",(T2-T1))

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

value = 100 * np.mean((predicted_label == validation_label).astype(float))

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset


print('Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

pickle_obj = [feature_list, n_hidden, w1, w2, lambdaval]
file_name = "/home/anurag/Desktop/" + str(n_hidden) + "_" + str(lambdaval) + "_" +  'params.pickle'
pickle.dump(pickle_obj, open(file_name, 'wb'))
    