# Made by Marcus Jirwe for DD2424

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import pickle

# Below are functions for reading in the data (one to unpickle the labels and one to load in the batches. These are adapted from the library of 
# https://github.com/snatch59/load-cifar-10 

# Essentially all functions are adapted from code used in assignment 1, only being changed to fit the fact that we now have a hidden layer.

def unpickle(file):
    ''' 
    Unpickles a file (used for labels, in the assignment).
    
    Args: 
        file (str), the filename of the file to be unpickled

    Output: 
        unpickled_file, a dictionary of the unpickled file
    '''

    with open(file, 'rb') as fo:
        unpickled_file = pickle.load(fo, encoding='bytes')
    return unpickled_file

def load_batch(file):
    '''
    Unpickles and loads the data into the corresponding matrices, with dimensions as in the assignment text and some values normalised (see below).
    Edited from last assignment is the fact that the batch normalisation is performed inside of this function on the data in the same function as the data is loaded.

    Args: 
        file (str), filename of the file to be unpickled and split

    Output: 
        X, a matrix of the data (dimension DxN, with D being dimensionality of each image and N being the amount of images)
        Y, a matrix containing the one-hot representations of the labels for each image (dimension KxN, with K being the amount of labels (10))
        y, a vector of length N that contains the the label for each image. 
    '''

    with open(file, 'rb') as fo:
        unpickled_file = pickle.load(fo, encoding='bytes') # The unpickled file is a dictionary of the data

        # The transpose below is to conform to the dimensionality given in the assigment, since the default dimension of numpy matrices are the transpose
        # of the ones in matlab.

        X = (unpickled_file[b'data'] / 255).T # Here the values are normalised to get values between 0 and 1 as opposed to 0 and 255 (default)
        y = unpickled_file[b'labels']
        Y = (np.eye(10)[y]).T # To give an example, the label "0", would look like [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        mean_X = np.mean(X, axis = 1).reshape(-1, 1)
        std_X = np.std(X, axis = 1).reshape(-1, 1)

        X = (X - mean_X)/std_X

        return X, Y, y

def load_small_data():
    '''
    Loads a smaller subset of data as specified in exercise 1 of the assignment description. 
    data_1_batch is used as training data, data_2_batch as validation data and test_batch as test data.

    Args:
        None
    
    Output:
        X_train, data_1_batch loaded as training data
        Y_train, the corresponding one-hot repr. of labels
        y_train, the corresponding labels

        X_val, Y_val, y_val, X_test, Y_test, y_test, the same but for the validation and test sets respectively.
        labels, the label names for the data.
    '''

    X_train, Y_train, y_train = load_batch("cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = load_batch("cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = load_batch("cifar-10-batches-py/test_batch")

    labels = unpickle('cifar-10-batches-py/batches.meta')[ b'label_names']

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels

def load_all_data(val_amt = 1000):
    '''Loads all the available training data as specified to be used for the best found lambda value. All the batches are loaded and concatenated to 
    a large training set, aside from the specified 1000 (can be adjusted) data points that are used as the validation set. The test batch is the same as usual.

    Args:
        None
    
    Output:
        X_train, data_1_batch loaded as training data
        Y_train, the corresponding one-hot repr. of labels
        y_train, the corresponding labels

        X_val, Y_val, y_val, X_test, Y_test, y_test, the same but for the validation and test sets respectively.
        labels, the label names for the data.
    '''

    X_train0, Y_train0, y_train0 = load_batch("cifar-10-batches-py/data_batch_1")
    X_train1, Y_train1, y_train1 = load_batch("cifar-10-batches-py/data_batch_2")
    X_train2, Y_train2, y_train2 = load_batch("cifar-10-batches-py/data_batch_3")
    X_train3, Y_train3, y_train3 = load_batch("cifar-10-batches-py/data_batch_4")
    X_train4, Y_train4, y_train4 = load_batch("cifar-10-batches-py/data_batch_5")

    X_test, Y_test, y_test = load_batch("cifar-10-batches-py/test_batch")

    labels = unpickle('cifar-10-batches-py/batches.meta')[ b'label_names']

    X_train = np.concatenate((X_train0, X_train1, X_train2, X_train3, X_train4), axis = 1)
    Y_train = np.concatenate((Y_train0, Y_train1, Y_train2, Y_train3, Y_train4), axis = 1)
    y_train = np.concatenate((y_train0, y_train1, y_train2, y_train3, y_train4))

    X_val = X_train[:, -val_amt:]
    Y_val = Y_train[:, -val_amt:]
    y_val = y_train[-val_amt:]

    X_train = X_train[:, :-val_amt]
    Y_train = Y_train[:, :-val_amt]
    y_train = y_train[:-val_amt]

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels

def coarse_grid_search(X_train, Y_train, y_train, X_val, Y_val, y_val, seed = 0, l_min = -5, l_max = -1, values = 20):
    ''' Performs a coarse grid search for an optimal value of the regression parameter lambda in the interval (1e(l_min), 1e(l_max))

    Args:
        X_train, Y_train, y_train, X_val, Y_val, y_val, matrices that define the data of the training and validation set for training and model comparison.
        seed, a scalar which defines the argument to np.random.seed for parameter initialisation of the model for fair comparison of lambda values.
        l_min, a scalar which defines the lower bound of the log interval to search for a good parameter value of lambda
        l_max, a scalar which defines the upper bound of the log interval.
        values, a scalar determining how many different values that are to be tested, i.e., how many values to sample from the interval.

    Output:
        None, the lambda values and associated accuracies are printed to the console and are not returned/saved explicitly.
    '''

    l_vals = np.random.uniform(-5, -1, (values, )) # Samples the values for the log search.
    tens = 10*np.ones(values)

    lamdas = np.power(tens, l_vals) # Computes the actual lamdas to be used for training the different networks

    tr_accs = np.zeros(values)
    val_accs = np.zeros(values)

    for i in range(values):
        print('Value ' + str(i) + ' out of ' + str(values-1))
        np.random.seed(seed)
        model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])
        np.random.seed()
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_s = 900, n_epochs = 8, lamda = lamdas[i])
        tr_accs[i] = model.compute_accuracy(X_train, y_train)
        val_accs[i] = model.compute_accuracy(X_val, y_val)

    bestThreeArgs = np.argsort(val_accs)[-3:] # Finds the index of the three best validation accuracies, which then corresponds to the three best networks.

    for j in range(3):
        print('Lambda: ' + str(lamdas[bestThreeArgs[j]]))
        print('Training Accuracy: ' + str(tr_accs[bestThreeArgs[j]]))
        print('Validation Accuracy: ' + str(val_accs[bestThreeArgs[j]]))


def fine_grid_search(center, width, X_train, Y_train, y_train, X_val, Y_val, y_val, seed = 0, values = 20):
    ''' Performs a fine grid search around a good lambda value discovered in the coarse grid search, in the interval (center - width, center + width).
        Function is largely the same code-wise as the coarse search, but the interval is defined differently.

    Args:
        center, a scalar that defines the center of the interval to be searched. This would be the discovered good lambda value from the coarse grid search.
        width, a scalar denoting the lower and upper bounds around the center to be searched.
        X_train, Y_train, y_train, X_val, Y_val, y_val, matrices that define the data of the training and validation set for training and model comparison.
        seed, a scalar which defines the argument to np.random.seed for parameter initialisation of the model for fair comparison of lambda values.
        values, a scalar determining how many different values that are to be tested, i.e., how many values to sample from the interval.

    Output:
        None, the lambda values and associated accuracies are printed to the console and are not returned/saved explicitly.
    '''

    lamdas = np.random.uniform(center - width, center + width, (values, ))
    tr_accs = np.zeros(values)
    val_accs = np.zeros(values)

    for i in range(values):
        print('Value ' + str(i) + ' out of ' + str(values-1))
        np.random.seed(seed)
        model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])
        np.random.seed()
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_s = 900, n_epochs = 8, lamda = lamdas[i])
        tr_accs[i] = model.compute_accuracy(X_train, y_train)
        val_accs[i] = model.compute_accuracy(X_val, y_val)

    bestThreeArgs = np.argsort(val_accs)[-3:] # Finds the index of the three best validation accuracies, which then corresponds to the three best networks.

    for j in range(3):
        print('Lambda: ' + str(lamdas[bestThreeArgs[j]]))
        print('Training Accuracy: ' + str(tr_accs[bestThreeArgs[j]]))
        print('Validation Accuracy: ' + str(val_accs[bestThreeArgs[j]]))

def compare_gradients(ana_grads, num_grads):
    ''' Compares the matrices that is the analytical and numerical gradients by defining a relative difference as being the Frobenius norm of their difference.
        This is equivalent to flattening the matrix and taking the euclidean norm.
        Inspired by: https://towardsdatascience.com/coding-neural-network-gradient-checking-5222544ccc64
        The relative difference is defined as the norm of the difference between the gradients divided by the sum of their individual norms.

    Args:
        ana_grad, a dictionary containing the gradients for each layer of each kind of parameter. Uses the same structure as the parameter dictionary.
        num_grad, a dictionary containing the gradients using the same structure as above, but calculated numerically using centered difference.

    Output:
        None, the differences in the gradients are printed to the console using this function but nothing is explicitly returned.
    '''

    ana_norm = np.linalg.norm(ana_grads)
    num_norm = np.linalg.norm(num_grads)
    diff_norm = np.linalg.norm(ana_grads - num_grads)

    rel_diff = diff_norm/(ana_norm + num_norm + np.finfo(np.float).eps) # Epsilon because of possible numerical stability issues.

    print('The gradient relative difference is: ' + str(rel_diff))


class multiLinearClassifier():
    ''' A class representing the multi-linear classifier in the assignment. Methods for gradient descent and such are contained in the class. '''

    def __init__(self, K, D, M = 50, W1 = None, b1 = None, W2 = None, b2 = None):

        ''' Constructor for the class. The classifier now has a hidden layer. The numerical values for W1 and W2 are sampled as the assignment text specifies.
        This means that sampling from a gaussian with mean 0 and std sqrt(1/D) and sqrt(1/M) for the first and second layer, respectively. 
        b1 and b2 are instead initialised to contain only zeroes.

        Args: 
            K, the amount of labels. Used to set the dimensions of the weights and biases W & b
            D, The dimensionality of the data. Used to set the dimensions of the weights.
            M, the amount of units in the hidden layer.
            
            W1, a matrix of numerical weights from input to hidden layer. If not specified, the weights are created randomly according to assignment. Dim MxD
            b1, a matrix containing the biases for the classifier from input to hidden layer. If not specified, are created randomly as specified in assignment. Dim Mx1
            W2, a matrix of numerical weights from hidden layer to output. Dim KxM
            b2, a matrix containing biases for the classifier from hidden layer to output. Dim Kx1

        Output:
            self, a multiLinearClassifier object with the associated parameters, functions and methods.
        '''

        if W1 == None:
            self.W1 = np.random.normal(0, 1/np.sqrt(D), (M, D))
        else: 
            self.W1 = W1

        if b1 == None:
            self.b1 = np.zeros((M, 1))
        else:
            self.b1 = b1

        if W2 == None:
            self.W2 = np.random.normal(0, 1/np.sqrt(M), (K, M))
        else: 
            self.W2 = W2

        if b2 == None:
            self.b2 = np.zeros((K, 1))
        else:
            self.b2 = b2

        self.M = M

    def softmax(self, x):
        ''' Computes the softmax of an array and returns it.

        Args:
            x, an array of scalar values
        Output:
            s, a matrix of the same size with softmax:ed values
        '''

        s = np.exp(x - np.max(x, axis = 0)) / np.sum(np.exp(x - np.max(x, axis = 0)), axis = 0) # Equation (4) & (5)

        return s
    
    def relu(self, h):
        ''' Applies the ReLU activation on all the values of the given array

        Args:
            h, an array of scalar values
        Output:
            h, the matrix x but with the ReLU activation applied to the values in place.
        '''

        h[h<0]=0

        return h

    def evaluate_classifier(self, X):
        ''' Evaluate the classifier by computing the ReLU and Softmax of the data multiplied according to the computational grap in a forward pass.
        The subtraction of the max gives more numerical stability for the softmax, while giving the same output.
        Args:
            X, a matrix of data (dimension DxN)

        Output:
            p, a matrix of dimension KxN, containing probabilities from numerically stable softmax.
        '''

        s1 = np.matmul(self.W1, X) + self.b1 # Equation (1)
        h = self.relu(s1)
        s = np.matmul(self.W2, h) + self.b2 # Equation (3)
        p = self.softmax(s) # Problems with the softmax being "too sure"?

        return h, p

    def compute_cost(self, X, Y, lamda):
        ''' Compute the loss and cost according to equation (7) & (8). 
        To be noted that l refers to the loss, without any regularisation. J refers to the cost, which does include regularisation.

        Args:
            X, a matrix of data (dim DxN)
            Y, a matrix of one-hot representations of labels (dim KxN).
            lamda, a scalar than acts as the regularisation strength for the weight decay. A larger lamda means a greater regularisation (larger penalty term).

        Output:
            l, a scalar that denotes only the cross-entropy loss, excluding any regularisation.
            J, a scalar that denotes the loss of the model on the given data X. This loss is defined according to equation (7) in the assignment text.
        '''

        N = X.shape[1] # The amount of individual data points / images. This is represented by a "D" in the assignment text. 
                       # However, since N was used previously I have used it here as well.

        P = self.evaluate_classifier(X)[1]
        l = (1/N)*-np.sum(np.multiply(Y, np.log(P)))
        J = l + lamda*(np.sum(self.W1**2) + np.sum(self.W2**2)) # The first term is the cross entropy loss with one-hot representation.

        return l, J

    def compute_accuracy(self, X, y):
        ''' Computes the accuracy of the model on the given data with associated labels.

        Args:
            X, a matrix of data with dimension DxN
            y, a vector of length N with the labels of the data. 

        Output:
            accuracy, a scalar which is the proportion of correctly classified samples. #Correct/#Total
        '''

        # X loses all but one data point when function is called?
        N = X.shape[1] # The total amount of data points / images in the input

        predictions = np.argmax(self.evaluate_classifier(X)[1], axis = 0) # Picks out the greatest probability from the softmax, for each image.

        correct = predictions[predictions == np.array(y)].size

        accuracy = correct/N

        return accuracy

    def compute_gradients_num(self, X, Y, lamda = 0, h = 1e-5):
        ''' A numerical method calculation of the gradients. Adapted from the more precise matlab function. Uses the centered difference method.
        Default value for h taken as recommendation from assignment description.
        Essentially this function is an extension of the matlab function in python to work on the weights of both layers.

        Args:
            X, a matrix of data with dimension DxN.
            Y, a matrix of one-hot representations of labels. Dimension KxN. 
            lamda, a scalar which is the regularisation strength.
            h, a scalar which is the "width" of the interval used for the finite difference method.

        Output: 
            grad_W1, a matrix which is the gradient of the weights from input to hidden
            grad_b1, a matrix which is the gradient of the biases to hidden
            grad_W2, a matrix which is the gradient of the weights from hidden to output
            grad_b2, a matrix which is the gradient of the biases to output
        '''

        grad_W1 = np.zeros(self.W1.shape)
        grad_b1 = np.zeros(self.b1.shape)
        grad_W2 = np.zeros(self.W2.shape)
        grad_b2 = np.zeros(self.b2.shape)

        b_try = np.copy(self.b1)

        for i in range(len(b_try)):
            self.b1 = np.copy(b_try)
            self.b1[i] = self.b1[i] - h
            c1 = self.compute_cost(X, Y, lamda)[1]
            self.b1 = np.copy(b_try)
            self.b1[i] = self.b1[i] + h
            c2 = self.compute_cost(X, Y, lamda)[1]

            grad_b1[i] = (c2-c1)/(2*h)
        
        W_try = np.copy(self.W1)

        for i in np.ndindex(self.W1.shape):
            self.W1 = np.copy(W_try)
            self.W1[i] = self.W1[i] - h
            c1 = self.compute_cost(X, Y, lamda)[1]
            self.W1 = np.copy(W_try)
            self.W1[i] = self.W1[i] + h
            c2 = self.compute_cost(X, Y, lamda)[1]
            grad_W1[i] = (c2-c1)/(2*h)

        b_try = np.copy(self.b2)

        for i in range(len(b_try)):
            self.b2 = np.copy(b_try)
            self.b2[i] = self.b2[i] - h
            c1 = self.compute_cost(X, Y, lamda)[1]
            self.b2 = np.copy(b_try)
            self.b2[i] = self.b2[i] + h
            c2 = self.compute_cost(X, Y, lamda)[1]

            grad_b2[i] = (c2-c1)/(2*h)
        
        W_try = np.copy(self.W2)

        for i in np.ndindex(self.W2.shape):
            self.W2 = np.copy(W_try)
            self.W2[i] = self.W2[i] - h
            c1 = self.compute_cost(X, Y, lamda)[1]
            self.W2 = np.copy(W_try)
            self.W2[i] = self.W2[i] + h
            c2 = self.compute_cost(X, Y, lamda)[1]
            grad_W2[i] = (c2-c1)/(2*h)

        return grad_W1, grad_b1, grad_W2, grad_b2

    def compute_gradients(self, X, Y, lamda = 0):
        ''' Calculates the gradients of the weights and biases analytically, by using equation (10) and (11) from the assignment description with lecture notes.
        Equations and notation taken from lecture notes (L4, specifically.)

        Args:
            X, a matrix of data with dimension DxN.
            Y, a matrix of one-hot representations of labels. Dimension KxN. 
            lamda, a scalar which is the regularisation strength.
        
        Output: 
            grad_W1, a matrix which is the gradient of the weights from input to hidden
            grad_b1, a matrix which is the gradient of the biases to hidden
            grad_W2, a matrix which is the gradient of the weights from hidden to output
            grad_b2, a matrix which is the gradient of the biases to output
        '''
        
        N = X.shape[1] # Denoted as n_b in the lecture slides.
        K = Y.shape[0]
        M = self.M
        

        # Computation of the forward pass
        H, P = self.evaluate_classifier(X)

        # Computation of the backwards pass
        G = (P-Y)
        grad_W2 = (1/N)*np.matmul(G, H.T) + 2 * lamda * self.W2
        grad_b2 = (1/N)*np.matmul(G, np.diag(np.eye(N)))
        grad_b2 = np.reshape(grad_b2, (K, 1)) # Reshape grad_b2 into 2D array, same as b2

        G = np.matmul(self.W2.T, G)
        H[H <= 0] = 0 # ReLU

        G = np.multiply(G, H > 0) # Indicator function

        grad_W1 = (1/N)*np.matmul(G, X.T) + 2 * lamda * self.W1
        grad_b1 = (1/N)*np.matmul(G, np.diag(np.eye(N)))
        grad_b1 = np.reshape(grad_b1, (M, 1))

        return grad_W1, grad_b1, grad_W2, grad_b2


    def mini_batch_gradient_descent(self, X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min = 1e-5, eta_max = 1e-1, n_s = 800, n_epochs = 40, lamda = 0, plot = False, printing = False, test = False, X_test = None, y_test = None):
        ''' Performs training if the multi-linear classifier using mini batch gradient descent. Default paramters are taken from assignment text.

        Args:
            X_train, a matrix containing the training set portion of the data.
            Y_train, a matrix containing the one-hot representations of the training set portion of the data.
            y_train, a matrix containing the normal label representations of the training set portion of the data.
            X_val, a matrix containing the validation set portion of the data.
            Y_val, a matrix containing the one-hot representations of the validation set portion of the data.
            y_val, a matrix containing the normal label representations of the validation set portion of the data.
            n_batch, a scalar integer that determines the size (amount of examples) of each mini batch.
            eta_min, a scalar float determining the lower bound for the learning rate.
            eta_max, a scalar determining the upper bound for the learning rate.
            n_s, a scalar that determines the step size which defines the update scheme for the cyclic learning rate.
            n_epochs, a scalar determining how many times we train on all the data. (Training once on all mini-batches is one epoch.)
            lamda, a scalar float determining the strength of the regularisation.

            plot, a bool that determines if the results are to be plotted or not. If plotting is not necessary, cuts down on unnecessary computations.
            print, a bool that determines if the accuracy of the trained model should be printed to the console. Cuts down on calculating accuracy if unnecessary.
            test, a bool that determines if the test accuracy should also be computed and printed. When doing the grid search this is unnecessary.
            X_test, a matrix containing test data
            y_test, a matrix containing the associated labels of the test data. Only necessary if performance of network is to be printed.

        
        Output:
            None, the algorithm changes the parameters in place as they are class variables. The plots are also done in the method.
        '''

        if plot:
            training_loss = np.zeros(n_epochs)
            training_cost = np.zeros(n_epochs)
            training_acc = np.zeros(n_epochs)
            validation_loss = np.zeros(n_epochs)
            validation_cost = np.zeros(n_epochs)
            validation_acc = np.zeros(n_epochs)

        N = int(X_train.shape[1]/n_batch) # The amount of batches that we have.

        t = 0 # Initialisation of the parameter of the same notation specified for the cyclic learning rate in equation (14).

        eta_t = eta_min # We intialise the learning rate to be the minimum, as shown in fig 2 in the assignment text.

        for n in range(n_epochs):
            #print("Epoch: " + str(n))
        
            order = np.arange(N) # Shuffles the indices of the mini-batches for training. A new shuffle is done very epoch.
            order = np.random.permutation(order)

            for j in order:
                j_start = j*n_batch
                j_end = (j+1)*n_batch
                X_batch = X_train[:, j_start:j_end]

                Y_batch = Y_train[:, j_start:j_end]

                grad_W1, grad_b1, grad_W2, grad_b2 = self.compute_gradients(X_batch, Y_batch, lamda)

                self.W1 += -eta_t * grad_W1
                self.b1 += -eta_t * grad_b1
                self.W2 += -eta_t * grad_W2
                self.b2 += -eta_t * grad_b2

                if (0 <= t <= n_s):
                    eta_t = eta_min + (t/n_s)*(eta_max - eta_min)

                elif (n_s <= t <= 2*n_s):
                    eta_t = eta_max - ((t - n_s)/n_s)*(eta_max - eta_min)

                t += 1 # t is incremented after each update of the learning rate.
                t = t % (2*n_s) # By performing the modulo operation t gets reset back to zero, meaning we can always consider l = 0 for the equations (14) & (15). 
                # By keeping t within the interval [0, 2*n_s) updates of the learning rate are more straight forward.

            if plot: # Execution is of an epoch is almost twice as fast for runs without plot if calculation of loss not done unless necessary (for plotting).
                training_loss[n], training_cost[n] = self.compute_cost(X_train, Y_train, lamda)
                validation_loss[n], validation_cost[n] = self.compute_cost(X_val, Y_val, lamda)
                training_acc[n] = self.compute_accuracy(X_train, y_train)
                validation_acc[n] = self.compute_accuracy(X_val, y_val)


        if plot:
            rows = 1
            cols = 3
            gs = gridspec.GridSpec(rows, cols)
            epochs = np.arange(n_epochs)

            fig = plt.figure(figsize=(10,3))
            
            ax = fig.add_subplot(gs[0])
            ax.plot(epochs, training_cost, label = 'Tr. Cost')
            ax.plot(epochs, validation_cost, label = 'Val. Cost')
            ax.legend()
            ax.set(xlabel='Epochs', ylabel = 'Cost')

            ax = fig.add_subplot(gs[1])
            ax.plot(epochs, training_loss, label = 'Tr. Loss')
            ax.plot(epochs, validation_loss, label = 'Val. Loss')
            ax.legend()
            ax.set(xlabel='Epochs', ylabel = 'Loss')

            ax = fig.add_subplot(gs[2])
            ax.plot(epochs, training_acc, label = 'Tr. Acc.')
            ax.plot(epochs, validation_acc, label = 'Val. Acc.')
            ax.legend()
            ax.set(xlabel='Epochs', ylabel = 'Accuracy')

            plt.tight_layout()
            plt.show()

        if printing:
            tr_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)
            if test:
                test_acc = self.compute_accuracy(X_test, y_test)

            print('Training Accuracy: ' + str(np.around(tr_acc, decimals=4)))
            print('Validation Accuracy: ' + str(np.around(val_acc, decimals = 4)))
            if test:
                print('Test Accuracy: ' + str(np.around(test_acc, decimals = 4)))



if __name__ == '__main__':

    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_small_data()

    # Testing if the numerical and analytical gradients are sufficiently close.
        
    model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])

    print('Calculating the analytical gradients...')

    grad_W1_test, grad_b1_test, grad_W2_test, grad_b2_test = model.compute_gradients(X_train[:, :2], Y_train[:, :2], lamda=0)

    print('Calculating the numerical gradients...')

    grad_W1_test_num, grad_b1_test_num, grad_W2_test_num, grad_b2_test_num= model.compute_gradients_num(X_train[:, :2], Y_train[:, :2], lamda=0)

    print('Comparing W1...')
    compare_gradients(grad_W1_test, grad_W1_test_num)
    print('Comparing b1...')
    compare_gradients(grad_b1_test, grad_b1_test_num)
    print('Comparing W2...')
    compare_gradients(grad_W2_test, grad_W2_test_num)
    print('Comparing b2...')
    compare_gradients(grad_b2_test, grad_b2_test_num)
    
    

    # Repliacate figure 3 from the assignment text
    '''
    
    model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])

    # We have 10000 samples in the one batch being trained on. Each mini-batch is 100 samples. This means we perform 100 update steps per epoch. Since one cycle is 1000 update steps, (n_s = 500)10 epochs is necessary.
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 500, n_epochs=10, lamda=0.01, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    
    '''

    # Replicate figure 4 from the assignment text
    
    '''

    model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])

    # One cycle of the learning rates now equates to performing 800*12 = 1600 updates steps. This means we need 4800 steps total for three cycles. Since the size of our batch is 100, that is equivalent to 48 epochs of training.
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 800, n_epochs=48, lamda=0.01, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)

    '''

    # Load in all the data to be used for the grid search.

    #X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data(5000) # 45k training points and 5k validation points

    # Perform a course grid search in the specified log interval

    '''

    n = X_train.shape[1]
    n_batch = 100
    n_s = int(2*np.floor(n/n_batch)) # As specified in the assignment text. n_s = 900.
    # This means that one learning rate cycle should correspond to 1800 update steps (meaning 2 cycles should be 8 epochs since we have 450 mini-batches 3600/450 = 8).

    coarse_grid_search(X_train, Y_train, y_train, X_val, Y_val, y_val, seed = 0, l_min = -5, l_max = -1, values = 50)
    
    '''

    # Perform a fine grid search in the specified interval (center - width, center + width) where center is a previously found lambda value.

    '''

    n = X_train.shape[1]
    n_batch = 100
    n_s = int(2*np.floor(n/n_batch)) # As specified in the assignment text. n_s = 900.

    # A search should be performed around each of the three best settings found in the coarse grid search, to be sure.

    best_coarse_lamdas = [0.004552705350104431, 0.003190465556971795, 0.002792580468572581]

    width = 0.001/4 # Chosen to be of the same magnitude as all the best lambdas from the coarse search. 

    for coarse_lam in best_coarse_lamdas: # For each of the fine grid searches the best 3 values and accuracies are reported.
        fine_grid_search(coarse_lam, width, X_train, Y_train, y_train, X_val, Y_val, y_val, values = 50)

    '''

    # Performs 3 cycles of training with the best found lambda value through the course followed by fine search. To generate the plots.

    '''

    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data(5000)

    n = X_train.shape[1]
    n_batch = 100
    n_s = int(2*np.floor(n/n_batch)) # As specified in the assignment text. n_s = 980. One cycle is therefore 1960 update steps.
    # Since we want to go for three update cycles that would be 5880 update steps. As training data we have 49000 data. With a batch size of 100, this equates to 490 mini batches.
    # Which of course means 490 update steps to cover all minibatches, meaning one epoch. The total amount of epochs is thus 5880/490 = 12.

    best_lamda = 0.003330827626680218
    model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = n_s, n_epochs=12, lamda=best_lamda, plot=True, printing = True)

    '''

    # Performs a more statistical analysis of the performance of the network for the best lambda by finding the mean and standard deviation of the accuracies.
    # Since the network will be somewhat susceptible to the initialisation of the parameters. 10 runs is performed with different initialisations.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data(5000)

    n = X_train.shape[1]
    n_batch = 100
    n_s = int(2*np.floor(n/n_batch)) # n_s = 980

    best_lamda = 0.003330827626680218
    model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])
    runs = 30
    training_accuracies = np.zeros(runs)
    validation_accuracies = np.zeros(runs)
    test_accuracies = np.zeros(runs)

    for i in range(runs):
        print('Run ' + str(i) + ' out of ' + str(runs-1))
        model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0])
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = n_s, n_epochs=12, lamda=best_lamda, plot=False, printing = False, test=True, X_test = X_test, y_test = y_test)

        training_accuracies[i] = model.compute_accuracy(X_train, y_train)
        validation_accuracies[i] = model.compute_accuracy(X_val, y_val)
        test_accuracies[i] = model.compute_accuracy(X_test, y_test)

    print('Training accuracy: ' + str(np.around(np.mean(training_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(training_accuracies), decimals = 4)))
    print('Validation accuracy: ' + str(np.around(np.mean(validation_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(validation_accuracies), decimals = 4)))
    print('Test accuracy: ' + str(np.around(np.mean(test_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(test_accuracies), decimals = 4)))
    #np.around(tr_acc, decimals=4))
    '''