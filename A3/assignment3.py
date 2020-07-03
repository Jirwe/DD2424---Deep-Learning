# Made by Marcus Jirwe for DD2424

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import pickle

# Below are functions for reading in the data (one to unpickle the labels and one to load in the batches. These are adapted from the library of 
# https://github.com/snatch59/load-cifar-10 

# Essentially all functions are adapted from code used in assignment 2, only being changed to fit the fact that we now have k hidden layers.
# Of course, structures had to be rewritten. Since I could not just hard code the backward pass, a dictionary had to be used to keep track of all the weight matrices.

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

def coarse_grid_search(X_train, Y_train, y_train, X_val, Y_val, y_val, seed = 0, l_min = -5, l_max = -1, values = 20, layer_shapes = [(50, 3072), (50, 50), (10, 50)]):
    ''' Performs a coarse grid search for an optimal value of the regression parameter lambda in the interval (1e(l_min), 1e(l_max))

    Args:
        X_train, Y_train, y_train, X_val, Y_val, y_val, matrices that define the data of the training and validation set for training and model comparison.
        seed, a scalar which defines the argument to np.random.seed for parameter initialisation of the model for fair comparison of lambda values.
        l_min, a scalar which defines the lower bound of the log interval to search for a good parameter value of lambda
        l_max, a scalar which defines the upper bound of the log interval.
        values, a scalar determining how many different values that are to be tested, i.e., how many values to sample from the interval.
        layer_shapes, a list of tuples specifying the architecture of the models to be trained.

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
        model = multiLinearClassifier(layer_shapes, batch_normalisation = True)
        np.random.seed()
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_s = 2250, n_epochs = 19, lamda = lamdas[i])
        tr_accs[i] = model.compute_accuracy(X_train, y_train)
        val_accs[i] = model.compute_accuracy(X_val, y_val)

    bestThreeArgs = np.argsort(val_accs)[-3:] # Finds the index of the three best validation accuracies, which then corresponds to the three best networks.

    for j in range(3):
        print('Lambda: ' + str(lamdas[bestThreeArgs[j]]))
        print('Training Accuracy: ' + str(tr_accs[bestThreeArgs[j]]))
        print('Validation Accuracy: ' + str(val_accs[bestThreeArgs[j]]))


def fine_grid_search(center, width, X_train, Y_train, y_train, X_val, Y_val, y_val, seed = 0, values = 20, layer_shapes = [(50, 3072), (50, 50), (10, 50)]):
    ''' Performs a fine grid search around a good lambda value discovered in the coarse grid search, in the interval (center - width, center + width).
        Function is largely the same code-wise as the coarse search, but the interval is defined differently.

    Args:
        center, a scalar that defines the center of the interval to be searched. This would be the discovered good lambda value from the coarse grid search.
        width, a scalar denoting the lower and upper bounds around the center to be searched.
        X_train, Y_train, y_train, X_val, Y_val, y_val, matrices that define the data of the training and validation set for training and model comparison.
        seed, a scalar which defines the argument to np.random.seed for parameter initialisation of the model for fair comparison of lambda values.
        values, a scalar determining how many different values that are to be tested, i.e., how many values to sample from the interval.
        layer_shapes, a list of tuples specifying the architecture of the models to be trained.

    Output:
        None, the lambda values and associated accuracies are printed to the console and are not returned/saved explicitly.
    '''

    lamdas = np.random.uniform(center - width, center + width, (values, ))
    tr_accs = np.zeros(values)
    val_accs = np.zeros(values)

    for i in range(values):
        print('Value ' + str(i) + ' out of ' + str(values-1))
        np.random.seed(seed)
        model = multiLinearClassifier(layer_shapes, batch_normalisation=True)
        np.random.seed()
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_s = 2250, n_epochs = 19, lamda = lamdas[i])
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
        ana_grads, a dictionary containing the gradients for each layer of each kind of parameter. Uses the same structure as the parameter dictionary.
        num_grads, a dictionary containing the gradients using the same structure as above, but calculated numerically using centered difference.

    Output:
        None, the differences in the gradients are printed to the console using this function but nothing is explicitly returned.
    '''

    # Loop through all the different keys 

    for param in ana_grads: # Loops through the different parameters, like 'W'
        for l in range(len(ana_grads[param])): # Loops through the weights for each layer.
            #print('Analytical: ' + str(ana_grads[param][l]))
            #print('Numerical: ' + str(num_grads[param][l]))
            #print('Grad dim: ' + str(np.shape(ana_grads[param][l])))
            #print('Param dim: ' + str(np.shape(model.params_dict[param][l])))
            ana_norm = np.linalg.norm(ana_grads[param][l])
            num_norm = np.linalg.norm(num_grads[param][l])
            diff_norm = np.linalg.norm(ana_grads[param][l] - num_grads[param][l])

            rel_diff = diff_norm/(ana_norm + num_norm + np.finfo(np.float).eps) # Epsilon because of possible numerical stability issues.

            print('The gradient relative difference for ' + param + ' for layer ' + str(l+1) + ': ' + str(rel_diff))





class multiLinearClassifier():
    ''' A class representing the multi-linear classifier in the assignment. Methods for gradient descent and such are contained in the class. '''

    def __init__(self, layer_shapes, batch_normalisation = False, alpha = 0.9, He = True, sigma = 0):

        ''' Constructor for the class. The classifier now has k hidden layers. The numerical values for the parameters are sampled using He-initialisation
        or simply using the same standard deviation, when testing sensitity to initialisations. Parameters are stored in a dictionary, inspired by:
        https://towardsdatascience.com/coding-neural-network-gradient-checking-5222544ccc64

        Args: 
            layer_shapes, a list of tuples that contains the shapes for the weight matrices of each layer in the network.
            batch_normalisation, a boolean that specifies whether batch normalisation should be used or not.
            alpha, a scalar that determines the 'Decay rate' of the moving average.

        Output:
            self, a multiLinearClassifier object with the associated parameters, functions and methods.
        '''

        self.layer_shapes = layer_shapes
        self.k = len(layer_shapes) - 1 # The index of the final layer. If three layers, should be 2.
        self.batch_normalisation = batch_normalisation
        self.alpha = alpha
        # Initialises the lists which will contain all the layers' weights and biases and such as empty lists.
        self.W = []
        self.b = []
        self.gamma = [] 
        self.beta = [] 
        self.mu_av = []
        self.v_av = []

        for shape in layer_shapes:
            W, b, gamma, beta, mu_av, v_av = self.param_initialisation(shape, He = He, sigma = sigma)
            self.W.append(W)
            self.b.append(b)
            self.gamma.append(gamma)
            self.beta.append(beta)
            self.mu_av.append(mu_av)
            self.v_av.append(v_av)

        # By not including the parameters gamma and beta when no batch normalisation is done we cut down on unnecessary gradient computations.
        if not self.batch_normalisation:
            self.params_dict = {'W' : self.W, 'b' : self.b}
        
        else:
            self.params_dict = {'W' : self.W, 'b' : self.b, 'gamma' : self.gamma, 'beta' : self.beta}

        
        
    def param_initialisation(self, shape, He = True, sigma = 0):
        ''' Performs parameter initialisation for a layer in the neural network. Uses He-initialisation unless specified not to (to test sensitivity)

        Args:
            shape, a tuple that specifies the dimensions of the weight matrix for a layer.
            He, a boolean that specifies whether He initialisation should be used or not.
            sigma, the standard deviation that should be used if not using He-initialisation.
        Output:
            W, the weight matrix for a layer of the network. Has the shape specified by the tuple of the same name.
            b, the bias vector for a layer of the network. Has the shape (shape[0], 1).
            gamma, the scaling vector for a layer of the network. Specifies the scaling there for batch normalisation of that layer.
            beta, the shift vector for a layer of the network which specifies the shift there for batch normalisation.
            mu_av, the moving average of the mean.
            v_av, the moving average of the variance.
        '''

        if He:
            sigma = 2/np.sqrt(shape[1])
        else:
            sigma = 1e-1 # Test different values here

        vec_shape = (shape[0], 1) # The shape of everything but the weight matrix. They are all made into 2D arrays for potential matrix multiplications.

        W = np.random.normal(0, sigma, shape)
        b = np.zeros(vec_shape)
        gamma = np.ones(vec_shape)
        beta = np.zeros(vec_shape)
        mu_av = np.zeros(vec_shape)
        v_av = np.zeros(vec_shape)

        return W, b, gamma, beta, mu_av, v_av

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

    def evaluate_classifier(self, X, test_time = False):
        ''' Evaluate the classifier by computing the ReLU and Softmax of the data multiplied according to the computational graph in a forward pass.
        The subtraction of the max gives more numerical stability for the softmax, while giving the same output.
        Args:
            X, a matrix of data (dimension DxN)
            test_time, a boolean that determines if batch normalisation (if applicable) should use equations (5)-(11) (True) or the corresponding training equations (38) & (39)

        Output:
            p, a matrix of dimension KxN, containing probabilities from numerically stable softmax.
        '''

        N = X.shape[1]
        s = np.copy(X) # A copy of the data is assigned to the parameter s, which will be changed through the forward pass.

        if not self.batch_normalisation:

            H = [] # Makes a list for the ReLU values of all the layers.

            # How to iterate over two lists at the same time and receive and index sources from: https://www.saltycrane.com/blog/2008/04/how-to-use-pythons-enumerate-and-zip-to/
            for l, (W, b) in enumerate(zip(self.W, self.b)):

                if l == 0:
                    s = self.relu(np.matmul(W, X) + b)
                    H.append(s)


                elif l < self.k: # If we are not at the last layer we apply the ReLU activation.
                    s = self.relu(np.matmul(W, s)  + b)
                    H.append(s)

                else: # If we are at the final layer we instead apply the softmax activation.
                    P = self.softmax(np.matmul(W, s) + b)

        
            return H, P

        else:
            # Initialises all the lists to be returned as empty lists.
            H = []
            S = []
            S_hat = [] 
            mu_l = []
            v_l = [] 

            for l, (W, b, gamma, beta, mu_av, v_av) in enumerate(zip(self.W, self.b, self.gamma, self.beta, self.mu_av, self.v_av)):

                if l == 0:
                    s = np.matmul(W, X) + b
                    

                else:
                    s = np.matmul(W, s) + b # Equation (12)

                if l < self.k:
                    S.append(s)
                    if test_time:
                        s = (s - mu_av) / np.sqrt(v_av + np.finfo(np.float).eps) # Equation (11)

                    else:
                        mu = np.mean(s, axis = 1) # Equation (13)
                        mu = np.reshape(mu, (-1, 1)) # Reshape so that it will broadcast.
                        mu_l.append(mu)
                        v = np.var(s, axis = 1)*(N-1)/N # Does the compensation mentioned in the assignment text. Equation (14)
                        v = np.reshape(v, (-1, 1))
                        v_l.append(v)

                        self.mu_av[l] = self.alpha * mu_av + (1 - self.alpha) * mu # Equation (38)
                        self.v_av[l] = self.alpha * v_av + (1 - self.alpha) * v # Equation (39)

                        s = (s - mu) / np.sqrt(v + np.finfo(np.float).eps) # Equation (11)

                    S_hat.append(s) # Equation (6)
                    s = self.relu(np.multiply(gamma, s) + beta) # Equation (7) + (8)
                    H.append(s) # Same as above

                else:
                    P = self.softmax(s)

            return H, P, S, S_hat, mu_l, v_l

    def compute_cost(self, X, Y, lamda, test_time = False):
        ''' Compute the loss and cost according to equation (7) & (8). 
        To be noted that l refers to the loss, without any regularisation. J refers to the cost, which does include regularisation.

        Args:
            X, a matrix of data (dim DxN)
            Y, a matrix of one-hot representations of labels (dim KxN).
            lamda, a scalar than acts as the regularisation strength for the weight decay. A larger lamda means a greater regularisation (larger penalty term).
            test_time, a bool that determines whether the evaluation of the classifier should use the testing equations for batch norm. (if applicable)

        Output:
            l, a scalar that denotes only the cross-entropy loss, excluding any regularisation.
            J, a scalar that denotes the loss of the model on the given data X. This loss is defined according to equation (7) in the assignment text.
        '''

        N = X.shape[1] # The amount of individual data points / images. This is represented by a "D" in the assignment text. 
                       # However, since N was used previously I have used it here as well.


        if not self.batch_normalisation:
            P = self.evaluate_classifier(X)[1]
            
        else:
            P = self.evaluate_classifier(X, test_time = test_time)[1]
        
        l = (1/N)*-np.sum(np.multiply(Y, np.log(P)))
        complexity = 0
        for W in self.W:
            complexity += np.sum(W**2)

        J = l + lamda * complexity # The first term is the cross entropy loss with one-hot representation.

        return l, J

    def compute_accuracy(self, X, y, test_time = False):
        ''' Computes the accuracy of the model on the given data with associated labels.

        Args:
            X, a matrix of data with dimension DxN
            y, a vector of length N with the labels of the data.
            test_time, a bool that determines whether test-version of eqations for batch_normalisation should be used. 

        Output:
            accuracy, a scalar which is the proportion of correctly classified samples. #Correct/#Total
        '''

        # X loses all but one data point when function is called?
        N = X.shape[1] # The total amount of data points / images in the input

        if not self.batch_normalisation:
            predictions = np.argmax(self.evaluate_classifier(X)[1], axis = 0)

        else:
            predictions = np.argmax(self.evaluate_classifier(X, test_time)[1], axis = 0) # Picks out the greatest probability from the softmax, for each image.

        correct = predictions[predictions == np.array(y)].size

        accuracy = correct/N

        return accuracy

    def compute_gradients_num(self, X, Y, lamda = 0, h = 1e-5):
        ''' A numerical method calculation of the gradients. Adapted from the more precise matlab function. Uses the centered difference method.
        Default value for h taken as recommendation from assignment description.
        Essentially this function is an extension of the matlab function in python to work on the weights of both layers.
        Inspiration for flattening/vectorising parameters for gradient calculations from: https://towardsdatascience.com/coding-neural-network-gradient-checking-5222544ccc64
        With numpys 'flatten' function taking the part of the vectorising function in the link.

        Args:
            X, a matrix of data with dimension DxN.
            Y, a matrix of one-hot representations of labels. Dimension KxN. 
            lamda, a scalar which is the regularisation strength.
            h, a scalar which is the "width" of the interval used for the finite difference method.

        Output: 
            gradients, a dictionary containing the gradients for the parameters of all the layers of the network.
        '''


        if not self.batch_normalisation:
            gradients = {'W' : [], 'b' : []}

        else:
            gradients = {'W' : [], 'b' : [], 'gamma' : [], 'beta' : []}

        for param in self.params_dict: # The iterator over the different parameters (keys) in the parameter dictionary
            for l in range(len(self.W)): # The iterator over the layers in the network.
                gradients[param].append(np.zeros(self.params_dict[param][l].shape)) # Different layers' params can have different shapes. Corresponding gradient has same shape.
                param_try = np.copy(self.params_dict[param][l])
                for i in range(len(self.params_dict[param][l].flatten())): # A one dimensional iterator over the elements of the current parameter.
                    self.params_dict[param][l] = np.copy(param_try)
                    self.params_dict[param][l].flat[i] = self.params_dict[param][l].flat[i] - h
                    c1 = self.compute_cost(X, Y, lamda)[1]
                    self.params_dict[param][l] = np.copy(param_try)
                    self.params_dict[param][l].flat[i] = self.params_dict[param][l].flat[i] + h
                    c2 = self.compute_cost(X, Y, lamda)[1]
                    self.params_dict[param][l] = np.copy(param_try)

                    gradients[param][l].flat[i] = (c2 - c1) / (2*h)

        return gradients

    def compute_gradients(self, X, Y, lamda = 0):
        ''' Calculates the gradients of the weights and biases analytically, by using equation (10) and (11) from the assignment description with lecture notes.
        Equations and notation taken from lecture notes (L4, specifically.)

        Args:
            X, a matrix of data with dimension DxN.
            Y, a matrix of one-hot representations of labels. Dimension KxN. 
            lamda, a scalar which is the regularisation strength.
        
        Output: 
            gradients, a dictionary containing the gradients for the parameters of all the layers of the network.
        '''
        
        N = X.shape[1] # Denoted as n_b in the lecture slides.

        k = self.k

        if not self.batch_normalisation:
            gradients = {'W' : [], 'b' : []}
            for W, b in zip(self.W, self.b):
                gradients['W'].append(np.zeros(W.shape))
                gradients['b'].append(np.zeros(b.shape))

            H_batch, P_batch = self.evaluate_classifier(X) # The forward pass

            G_batch = (P_batch - Y) # Start of the backward pass

            # Loops backwards through the layers propagating G & H and calculating the gradients

            for l in range(k, 0, -1): # Make a special case for l = 0 since no ReLU present there.
                gradients['W'][l] = (1/N)*np.matmul(G_batch, H_batch[l-1].T) + 2 * lamda * self.W[l] # Equation (22)
                gradients['b'][l] = (1/N)*np.matmul(G_batch, np.diag(np.eye(N))) # Equation (22)
                gradients['b'][l] = np.reshape(gradients['b'][l], self.b[l].shape) # replace with -1?

                G_batch = np.matmul(self.W[l].T, G_batch) # Equation (23)
                H_batch[l-1][H_batch[l-1] <= 0] = 0 # ReLU
                G_batch = np.multiply(G_batch, H_batch[l-1] > 0) # Equation (24)

            # Special case for l = 0 because we do not need to propagate G_batch further.
            gradients['W'][0] = (1/N)*np.matmul(G_batch, X.T) + 2 * lamda * self.W[0]
            gradients['b'][0] = (1/N)*np.matmul(G_batch, np.diag(np.eye(N)))
            gradients['b'][0] = np.reshape(gradients['b'][0], self.b[0].shape)

        else: # Write code for the batch normalisation of the gradient calculation
            gradients = {'W' : [], 'b' : [], 'gamma' : [], 'beta' : []}
            for W, b, gamma, beta in zip(self.W, self.b, self.gamma, self.beta):
                gradients['W'].append(np.zeros(W.shape))
                gradients['b'].append(np.zeros(b.shape))
                gradients['gamma'].append(np.zeros(gamma.shape))
                gradients['beta'].append(np.zeros(beta.shape))

            H_batch, P_batch, S_batch, S_hat_batch, mu_batch, v_batch = self.evaluate_classifier(X) # Forward pass with batch normalisation.

            G_batch = (P_batch - Y) # Start of the backward pass with batch normalisation.

            # Special case for l = k since the gradients for gamma and beta will be one element shorter (same loop as assignment text)
            gradients['W'][k] = (1/N) * np.matmul(G_batch, H_batch[k-1].T) + 2 * lamda * self.W[k]
            gradients['b'][k] = (1/N) * np.matmul(G_batch, np.diag(np.eye(N)))
            gradients['b'][k] = np.reshape(gradients['b'][k], self.b[k].shape)

            G_batch = np.matmul(self.W[k].T, G_batch) # Propagate the G_batch backwards to the k-1 layer.

            H_batch[k-1][H_batch[k-1] <= 0 ] = 0 # Equation (24)

            G_batch = np.multiply(G_batch, H_batch[k-1] > 0) # Equation (24)

            for l in range(k-1, 0, -1): 
                gradients['gamma'][l] = (1/N) * np.matmul(np.multiply(G_batch, S_hat_batch[l]), np.diag(np.eye(N))) # Equation (25)
                gradients['gamma'][l] = np.reshape(gradients['gamma'][l], self.gamma[l].shape)
                gradients['beta'][l] = (1/N) * np.matmul(G_batch, np.diag(np.eye(N))) # Equation (25)
                gradients['beta'][l] = np.reshape(gradients['beta'][l], self.beta[l].shape)

                G_batch = np.multiply(G_batch, self.gamma[l]) # Equation (26)

                G_batch = self.batch_normalisation_back_pass(G_batch, S_batch[l], mu_batch[l], v_batch[l]) # A call to a function performing Equation (27)

                gradients['W'][l] = (1/N) * np.matmul(G_batch, H_batch[l-1].T) + 2 * lamda * self.W[l]
                gradients['b'][l] = (1/N) * np.matmul(G_batch, np.diag(np.eye(N)))
                gradients['b'][l] = np.reshape(gradients['b'][l], self.b[l].shape)

                # Propagate l backwards through the layers only if not on the first layer.

                G_batch = np.matmul(self.W[l].T, G_batch)
                H_batch[l-1][H_batch[l-1] <= 0 ] = 0 # Equation (24)
                G_batch = np.multiply(G_batch, H_batch[l-1] > 0) # Equation (24)

            gradients['gamma'][0] = (1/N) * np.matmul(np.multiply(G_batch, S_hat_batch[0]), np.diag(np.eye(N)))
            gradients['gamma'][0] = np.reshape(gradients['gamma'][0], self.gamma[0].shape)
            gradients['beta'][0] = (1/N) * np.matmul(G_batch, np.diag(np.eye(N)))
            gradients['beta'][0] = np.reshape(gradients['beta'][0], self.beta[0].shape)
            gradients['W'][0] = (1/N) * np.matmul(G_batch, X.T) + 2 * lamda * self.W[0]
            gradients['b'][0] = (1/N) * np.matmul(G_batch, np.diag(np.eye(N)))
            gradients['b'][0] = np.reshape(gradients['b'][0], self.b[0].shape)


        return gradients

    def batch_normalisation_back_pass(self, G_batch, S_batch, mu_batch, v_batch):
        ''' Computes the "BatchNormBackPass" function from the assignment description (Equation (27)), using equations (31)-(37).

        Args:
            G_batch, a matrix used to perform the backward pass and calculate gradients.
            S_batch, the s-scores of the batch at some layer in the backward pass.
            mu_batch, the mean of the batch at some layer in the backward pass.
            v_batch, the corresponding variance at some layer in the backward pass.

        Output:
            G_batch, the G_batch transformed by the use of batch normalisation.
        '''

        n = G_batch.shape[1]

        sigma1 = (v_batch + np.finfo(np.float).eps)**-0.5 # Equation (31)
        sigma2 = (v_batch + np.finfo(np.float).eps)**-1.5 # Equation (32)

        G1 = np.multiply(G_batch, sigma1) # Equation (33)
        G2 = np.multiply(G_batch, sigma2) # Equation (34)

        D = S_batch - mu_batch # Equation (35)

        c = np.sum(np.multiply(G2, D), axis = 1) # Equation (36). Summing can be performed here just as well as in the G_batch update below.

        c = np.reshape(c, (-1, 1))

        G_batch = G1 - (1/n) * np.sum(G1, axis = 1, keepdims = True) - (1/n) * np.multiply(D, c) # Equation (37). Multiplication with a transposed vector of ones unnecessary since we summed c above.

        return G_batch


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
            test, a bool that determines if the test accuracy should also be computed and printed. Also determines if the test-equations for batch normalisation should be used.
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

        t = 0 # Initialisation of the parameter of the same notation specified for the cyclic learning rate is assignment 2.

        eta_t = eta_min # We intialise the learning rate to be the minimum.

        for n in range(n_epochs):
            #print("Epoch: " + str(n))
        
            order = np.arange(N) # Shuffles the indices of the mini-batches for training. A new shuffle is done very epoch.
            order = np.random.permutation(order)

            for j in order:
                j_start = j*n_batch
                j_end = (j+1)*n_batch
                X_batch = X_train[:, j_start:j_end]

                Y_batch = Y_train[:, j_start:j_end]

                gradients = self.compute_gradients(X_batch, Y_batch, lamda) # Calculates all the gradients

                for param in gradients:
                    for l in range(len(gradients[param])):
                        self.params_dict[param][l] += -eta_t * gradients[param][l] # Updates the parameters at each layer.

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
                test_acc = self.compute_accuracy(X_test, y_test, test_time=True)

            print('Training Accuracy: ' + str(np.around(tr_acc, decimals=4)))
            print('Validation Accuracy: ' + str(np.around(val_acc, decimals = 4)))
            if test:
                print('Test Accuracy: ' + str(np.around(test_acc, decimals = 4)))



if __name__ == '__main__':

    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_small_data()

    # Testing the gradients for 2 layers.
    '''
    print('Checking the relative difference of the gradients for 2 layers.')
    layer_shapes_2_layers = [(50, 10), (10, 50)] # Testing gradients on ten elements of the data with architecture as recommended in the exercise for 2 layers.
    model = multiLinearClassifier(layer_shapes_2_layers)
    ana_grads = model.compute_gradients(X_train[:10, :10], Y_train[:10, :10]) # Using only ten elements for faster computing.
    num_grads = model.compute_gradients_num(X_train[:10, :10], Y_train[:10, :10])
    compare_gradients(ana_grads, num_grads)
    '''

    # Testing the gradients for 3 layers.
    '''
    print('Checking the relative difference of the gradients for 3 layers.')
    layer_shapes_3_layers = [(50, 10), (50, 50), (10, 50)] # Testing gradients on ten elements of the data with architecture as recommended in the exercise for 3 layers.
    model = multiLinearClassifier(layer_shapes_3_layers)
    ana_grads = model.compute_gradients(X_train[:10, :10], Y_train[:10, :10]) # Using only ten elements for faster computing
    num_grads = model.compute_gradients_num(X_train[:10, :10], Y_train[:10, :10])
    compare_gradients(ana_grads, num_grads) # For three layers, the worst relative difference seems to be in the order 1e-6. Most likely correct.
    '''

    # Testing the gradients for 4 layers.
    
    print('Checking the relative difference of the gradients for 4 layers.')
    layer_shapes_4_layers = shapes=[(50, 10), (50, 50), (50, 50), (10, 50)] # Testing gradients on ten elements of the data with recommended architecture.
    model = multiLinearClassifier(layer_shapes_4_layers, batch_normalisation=True) # When batch normalisation used, larger relative difference?
    ana_grads = model.compute_gradients(X_train[:10, :10], Y_train[:10, :10]) # Using only ten elements for faster computing.
    num_grads = model.compute_gradients_num(X_train[:10, :10], Y_train[:10, :10])
    compare_gradients(ana_grads, num_grads)
    
    
    # Check to see if able to replicate the results from assignment 2 with the general code. Using same hyperparameters.
    '''
    layer_shapes_2_layers = [(50, 3072), (10, 50)]
    model = multiLinearClassifier(layer_shapes_2_layers)
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 500, n_epochs=10, lamda=0.01, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    # Results seem to absolutely be comparable.
    '''

    # Check to see if able to replicate somewhat the 3-layer performance mentioned in the assignment description.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers = [(50, 3072), (50, 50), (10, 50)]
    model = multiLinearClassifier(layer_shapes_3_layers) # With an n_s of 2250, a learning rate cycle is 4500 update steps, e.g. 45 epochs (since we have 100 mini-batches per epoch). That means 9000/100 = 90 epochs for 2 cycles (small set) or around 18 (all data)
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    # I'm making the assumption here that considering the large n_s that I'm meant to train on all the data. Otherwise I seem to overfit to the smaller dataset (tr. accuracy ≈ .88). Results on all data is very much comparable.
    '''

    # Examine how the performance is altered when having more (9) layers without batch normalisation.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_9_layers = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    model = multiLinearClassifier(layer_shapes_9_layers) # With an n_s of 2250, a learning rate cycle is 4500 update steps, e.g. 45 epochs (since we have 100 mini-batches per epoch). That means 9000/100 = 90 epochs for 2 cycles (small set) or around 18 (all data)
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    # Performance is indeed worse than the corresponding amount of parameters being used in a 3-layer network, lending credit to the need for batch normalisation.
    '''

    # Testing the gradients for 9 layers with batch normalisation.
    '''
    print('Checking the relative difference of the gradients for 9 layers with batch normalisation.')
    layer_shapes_9_layers_bn = [(50, 10), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)] # Testing gradients on ten elements of the data with recommended architecture.
    model = multiLinearClassifier(layer_shapes_9_layers_bn, batch_normalisation = True)
    ana_grads = model.compute_gradients(X_train[:10, :10], Y_train[:10, :10]) # Using only ten elements for faster computing.
    num_grads = model.compute_gradients_num(X_train[:10, :10], Y_train[:10, :10])
    compare_gradients(ana_grads, num_grads)
    # Relative differences become greater as we reach earlier layers. Especially the relative error for the biases become rather large. Supposedly they cancel out anyway, though.
    '''

    # Examine how the performance of 3-layer net is affected by batch normalisation.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers_bn = [(50, 3072), (50, 50), (10, 50)]
    model = multiLinearClassifier(layer_shapes_3_layers_bn, batch_normalisation = True) # With an n_s of 2250, a learning rate cycle is 4500 update steps, e.g. 45 epochs (since we have 100 mini-batches per epoch). That means 9000/100 = 90 epochs for 2 cycles (small set) or around 18 (all data. 9000/490 = 18.32)
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    # Test accuracy seems to check out as being around the level written in the assignment text, only being 0.0065 or so below. This can be chalked up to a dependence on initialisation. Following tries included performances only 0.02 worse.
    '''

    # Examine how the performance is affected from the same structure of 9 layers but now using batch normalisation also.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_9_layers_bn = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    model = multiLinearClassifier(layer_shapes_9_layers_bn, batch_normalisation=True) # With an n_s of 2250, a learning rate cycle is 4500 update steps, e.g. 45 epochs (since we have 100 mini-batches per epoch). That means 9000/100 = 90 epochs for 2 cycles (small set) or around 18 (all data)
    model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    # Using the same hyperparameters but with batch normalisation seems to increase the validation accuracy by about 4 percentage points, and the test set by slightly less.
    '''

    # Perform a coarse grid search in the interval (-5, -1) after a good initial lambda.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers_bn = [(50, 3072), (50, 50), (10, 50)]
    coarse_grid_search(X_train, Y_train, y_train, X_val, Y_val, y_val, seed = 0, l_min = -5, l_max = -1, values = 50, layer_shapes = layer_shapes_3_layers_bn)
    '''

    # Perform a fine grid search in a small interval around the best 3 lambda-values found in the coarse search.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers_bn = [(50, 3072), (50, 50), (10, 50)]
    best_coarse_lamdas = [0.003759250129457098, 0.007624470110464359, 0.004179707368837784]
    width = 0.001/4
    for coarse_lam in best_coarse_lamdas: # For each of the fine grid searches the best 3 values and accuracies are reported.
        fine_grid_search(coarse_lam, width, X_train, Y_train, y_train, X_val, Y_val, y_val, values = 50, layer_shapes=layer_shapes_3_layers_bn)
    '''

    # Train a 3-layer network with the best lambda with three cycles of training ≈ 28 epochs. Should perhaps be tested more statistically, with 10 runs.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers_bn = [(50, 3072), (50, 50), (10, 50)]
    best_lamda = 0.0035476760480130176
    runs = 20
    training_accuracies = np.zeros(runs)
    validation_accuracies = np.zeros(runs)
    test_accuracies = np.zeros(runs)
    for i in range(runs):
        print('Run ' + str(i) + ' out of ' + str(runs-1))
        model = multiLinearClassifier(layer_shapes_3_layers_bn, batch_normalisation=True)
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=28, lamda=best_lamda, plot=False, printing = False, test = True, X_test = X_test, y_test = y_test)
        training_accuracies[i] = model.compute_accuracy(X_train, y_train)
        validation_accuracies[i] = model.compute_accuracy(X_val, y_val)
        test_accuracies[i] = model.compute_accuracy(X_test, y_test, test_time = True)
    print('Training accuracy: ' + str(np.around(np.mean(training_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(training_accuracies), decimals = 4)))
    print('Validation accuracy: ' + str(np.around(np.mean(validation_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(validation_accuracies), decimals = 4)))
    print('Test accuracy: ' + str(np.around(np.mean(test_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(test_accuracies), decimals = 4)))
    # Test accuracy turns out around 0.54 or so.
    '''

    # Train a 9-layer network with the best lambda (for 3-layers) with three cycles of training. Also test more statistically, with 10 runs.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_9_layers_bn = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    best_lamda = 0.0035476760480130176
    runs = 20
    training_accuracies = np.zeros(runs)
    validation_accuracies = np.zeros(runs)
    test_accuracies = np.zeros(runs)
    for i in range(runs):
        print('Run ' + str(i) + ' out of ' + str(runs-1))
        model = multiLinearClassifier(layer_shapes_9_layers_bn, batch_normalisation=True)
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=28, lamda=best_lamda, plot=False, printing = False, test = True, X_test = X_test, y_test = y_test)
        training_accuracies[i] = model.compute_accuracy(X_train, y_train) # Should still calculate the moving averages as if training?
        validation_accuracies[i] = model.compute_accuracy(X_val, y_val)
        test_accuracies[i] = model.compute_accuracy(X_test, y_test, test_time = True)
    print('Training accuracy: ' + str(np.around(np.mean(training_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(training_accuracies), decimals = 4)))
    print('Validation accuracy: ' + str(np.around(np.mean(validation_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(validation_accuracies), decimals = 4)))
    print('Test accuracy: ' + str(np.around(np.mean(test_accuracies), decimals = 4)) + ' +- ' + str(np.around(np.std(test_accuracies), decimals = 4)))
    '''

    # Train a 3-layer network with the recommended architecture with constant sigma across layers for initialisation testing without BN.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers = [(50, 3072), (50, 50), (10, 50)]
    sigmas = [1e-1, 1e-3, 1e-4] # The three sigmas that we should test.
    for sig in sigmas:
        print('Sigma = ' + str(sig))
        model = multiLinearClassifier(layer_shapes_3_layers, He = False, sigma = sig)
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    '''
    
    #  Train a 3-layer network with the recommended architecture and constant sigma across layers but with BN.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_3_layers_bn = [(50, 3072), (50, 50), (10, 50)]
    sigmas = [1e-1, 1e-3, 1e-4] # The three sigmas that we should test.
    for sig in sigmas:
        print('Sigma = ' + str(sig))
        model = multiLinearClassifier(layer_shapes_3_layers_bn, batch_normalisation = True, He = False, sigma = sig)
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    '''

    # Same as above but for a 9-layer network instead.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_9_layers = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    sigmas = [1e-1, 1e-3, 1e-4] # The three sigmas that we should test.
    for sig in sigmas:
        print('Sigma = ' + str(sig))
        model = multiLinearClassifier(layer_shapes_9_layers, He = False, sigma = sig)
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    '''

    # Same as above but with batch normalisation.
    '''
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, labels = load_all_data()
    layer_shapes_9_layers_bn = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    sigmas = [1e-1, 1e-3, 1e-4] # The three sigmas that we should test.
    for sig in sigmas:
        print('Sigma = ' + str(sig))
        model = multiLinearClassifier(layer_shapes_9_layers_bn, batch_normalisation = True, He = False, sigma = sig)
        model.mini_batch_gradient_descent(X_train, Y_train, y_train, X_val, Y_val, y_val, n_batch = 100, eta_min=1e-5, eta_max=1e-1, n_s = 2250, n_epochs=19, lamda=0.005, plot=True, printing = True, test = True, X_test = X_test, y_test = y_test)
    '''