# Made by Marcus Jirwe for DD2424

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import pickle

# Below are functions for reading in the data (one to unpickle the labels and one to load in the batches. These are adapted from the library of 
# https://github.com/snatch59/load-cifar-10 

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

        return X, Y, y

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

    def __init__(self, K, D, W = None, b = None):

        ''' Constructor for the class.

        Args: 
            K, the amount of labels. Used to set the dimensions of the weights and biases W & b
            D, the dimensionality of the data. Used to set the dimensions of the weights.
            W, a matrix of numerical weights. If not specified (like loading from file), the weights are created randomly according to assignment. Dim KxD
            b, a matrix containing the biases for the classifier. If not specified, are created randomly as specified in assignment. Dim Kx1
        '''

        if W == None:
            self.W = np.random.normal(0, 0.01, (K, D))
        else: 
            self.W = W

        if b == None:
            self.b = np.random.normal(0, 0.01, (K, 1))
        else:
            self.b = b

    def evaluate_classifier(self, X):
        ''' Evaluate the classifier by computing the softmax probabilities for some data. The subtraction of the max gives more numerical stability, while giving the same output.
        Args:
            X, a matrix of data (dimension DxN)

        Output:
            p, a matrix of dimension KxN, containing probabilities from numerically stable softmax.
        '''

        s = np.matmul(self.W, X) + self.b # Equation (1)
        p = np.exp(s - np.max(s, axis = 0)) / np.sum(np.exp(s - np.max(s, axis = 0)), axis = 0) # Equation (2) & (3). 

        return p

    def compute_cost(self, X, Y, lamda):
        ''' Compute the cost according to equation (5). 

        Args:
            X, a matrix of data (dim DxN)
            Y, a matrix of one-hot representations of labels (dim KxN).
            lamda, a scalar than acts as the regularisation strength for the weight decay. A larger lamda means a greater regularisation (larger penalty term).

        Output:
            J, a scalar that denotes the loss of the model on the given data X. This loss is defined according to equation (5) in the assignment text.
        '''

        N = X.shape[1] # The amount of individual data points / images. This is represented by a "D" in the assignment text. 
                       # However, since N was used previously I have used it here as well.

        P = self.evaluate_classifier(X)
        J = (1/N)*-np.sum(np.multiply(Y, np.log(P))) + lamda*np.sum(self.W**2) # The first term is the cross entropy loss with one-hot representation.

        return J

    def compute_accuracy(self, X, y):
        ''' Computes the accuracy of the model on the given data with associated labels.

        Args:
            X, a matrix of data with dimension DxN
            y, a vector of length N with the labels of the data. 

        Output:
            accuracy, a scalar which is the proportion of correctly classified samples. #Correct/#Total
        '''

        N = X.shape[1] # The total amount of data points / images in the input

        predictions = np.argmax(self.evaluate_classifier(X), axis = 0) # Picks out the greatest probability from the softmax, for each image.

        correct = predictions[predictions == np.array(y)].size

        accuracy = correct/N

        return accuracy

    def compute_gradients_num(self, X, Y, lamda = 0, h = 1e-6):
        ''' A numerical method calculation of the gradients. Adapted from the more precise matlab function. Uses the centered difference method.

        Args:
            X, a matrix of data with dimension DxN.
            Y, a matrix of one-hot representations of labels. Dimension KxN. 
            lamda, a scalar which is the regularisation strength.
            h, a scalar which is the "width" of the interval used for the finite difference method.

        Output: 
            grad_W, a matrix which is the gradient of the weights
            grad_b, a matrix which is the gradient of the biases
        '''

        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros(self.b.shape)

        b_try = np.copy(self.b)

        for i in range(len(b_try)):
            self.b = np.copy(b_try)
            self.b[i] = self.b[i] - h
            c1 = self.compute_cost(X, Y, lamda)
            self.b = np.copy(b_try)
            self.b[i] = self.b[i] + h
            c2 = self.compute_cost(X, Y, lamda)

            grad_b[i] = (c2-c1)/(2*h)
        
        W_try = np.copy(self.W)
        for i in np.ndindex(self.W.shape):
            self.W = np.copy(W_try)
            self.W[i] = self.W[i] - h
            c1 = self.compute_cost(X, Y, lamda)
            self.W = np.copy(W_try)
            self.W[i] = self.W[i] + h
            c2 = self.compute_cost(X, Y, lamda)
            grad_W[i] = (c2-c1)/(2*h)

        return grad_W, grad_b

    def compute_gradients(self, X, Y, lamda = 0):
        ''' Calculates the gradients of the weights and biases analytically, by using equation (10) and (11) from the assignment description with lecture notes.

        Args:
            X, a matrix of data with dimension DxN.
            Y, a matrix of one-hot representations of labels. Dimension KxN. 
            lamda, a scalar which is the regularisation strength.
        
        Output: 
            grad_W, a matrix which is the gradient of the weights
            grad_b, a matrix which is the gradient of the biases
        '''

        N = X.shape[1]
        K = Y.shape[0]

        P = self.evaluate_classifier(X)
        G = (P-Y) # Derivative of the cross entropy function and notation taken from the lecture notes of lecture 3.

        grad_W = (1/N)*np.matmul(G, X.T) + 2*lamda*self.W # X.T comes from the inner derivative of P w.r.t. W
        grad_b = (1/N)*np.matmul(G, np.diag(np.eye(N))) # However, in the case of biases, inner derivative is one (biases not multiplied with anything)
        grad_b = np.reshape(grad_b, (K, 1)) # Turn grad_b into a 2D array, same as b.

        return grad_W, grad_b


    def mini_batch_gradient_descent(self, X_train, Y_train, X_val, Y_val, n_batch = 100, eta = 0.001, n_epochs = 40, lamda = 0, plot = False):
        ''' Performs training if the multi-linear classifier using mini batch gradient descent. Default paramters are taken from assignment text.

        Args:
            X_train, a matrix containing the training set portion of the data
            Y_train, a matrix containing the one-hot representations of the training set portion of the data
            X_val, a matrix containing the validation set portion of the data
            Y_val, a matrix containing the one-hot representations of the validation set portion of the data
            n_batch, a scalar integer that determines how many mini-batches to split the training data into.
            eta, a scalar float determining the learning rate.
            n_epochs, a scalar determining how many times we train on all the data. (Training once on all mini-batches is one epoch.)
            lamda, a scalar float determining the strength of the regularisation.
            plot, a bool that determines if the loss curve is to be plotted or not. Turning this into an option cuts down on unneccesary computations when plotting is not needed, also.
        
        Output:
            None, the algorithm changes the parameters in place as they are class variables. The plots are also done in the method.
        '''

        training_loss = np.zeros(n_epochs)
        validation_loss = np.zeros(n_epochs)

        N = int(X_train.shape[1]/n_batch) # An integer representing the size of our mini-batches from the training data given the number of batches.

        for n in range(n_epochs):
        
            order = np.arange(n_batch)
            order = np.random.permutation(order)    # Randomly shuffles the index of the mini batches for training. New shuffle every epoch.

            for j in order:
                j_start = j*N
                j_end = (j+1)*N

                X_batch = X_train[:, j_start:j_end]
                Y_batch = Y_train[:, j_start:j_end]

                grad_W, grad_b = self.compute_gradients(X_batch, Y_batch, lamda)

                self.W += -eta * grad_W # Negative since we only calculated the gradient and want gradient descent.
                self.b += -eta * grad_b

            if plot: # Execution is of an epoch is almost twice as fast for runs without plot if calculation of loss not done unless necessary (for plotting).
                training_loss[n] = self.compute_cost(X_train, Y_train, lamda)
                validation_loss[n] = self.compute_cost(X_val, Y_val, lamda)


        if plot:

            epochs = np.arange(n_epochs)        
            plt.plot(epochs, training_loss, label = 'Training Loss')
            plt.plot(epochs, validation_loss, label = 'Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss on training/validation. \u03B7 = ' + str(eta) +  ', \u03BB = ' + str(lamda))
            plt.show()
        

    def show_images(self, setting, labels):
        ''' A function that saves the learnt weight matrices (10) into a subplot for a given parameter setting

        Args:
            setting, a string that is used for the filename to be saved (parameter settings from 0 to 3).
        '''

        rows = 2
        cols = 5
        gs = gridspec.GridSpec(rows, cols)

        fig = plt.figure(figsize=(10,10))

        for idx, mat in enumerate(self.W):
            mat = (mat - np.min(mat))/np.ptp(mat) # Normalisation sourced from https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range and assignment description.
            # This serves as the practical inverse of the batch normalisation we did before. Since for floats, entires in RGB images should be in [0, 1] interval.

            im = np.dstack((mat[0:1024].reshape(32, 32), mat[1024:2048].reshape(32, 32), mat[2048:].reshape(32, 32))) # Properly stacking the squished RGB image in the three channels


            byte_title = labels[idx]
            im_class = byte_title.decode('utf8')

            ax = fig.add_subplot(gs[idx])
            ax.imshow(im, interpolation='spline36') # add other interpolations?

            plt.axis('off')

            fig.subplots_adjust(top = 0.4, bottom = 0.1)

            ax.set_title(im_class)

        plt.savefig('images' + str(setting) + '.png', bbox_inches='tight')

            





if __name__ == '__main__':

    save = False
    load = False
    filename = 'model.npz'
    

    X_train, Y_train, y_train = load_batch("cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = load_batch("cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = load_batch("cifar-10-batches-py/test_batch")

    #print("X_train: " + str(X_train.shape)) # Checks out to be (3072, 10000)
    #print("Y_train: " + str(Y_train.shape)) # Checks out to be (10, 10000)
    #print("y_train: " + str(len(y_train))) # Checks out to be (10000, )

    mean_X = np.mean(X_train, axis = 1).reshape(-1, 1) # Turns the vector into a 2D array for broadcasting (for the batch normalisation)
    #print("mean_X: " + str(mean_X.shape))
    std_X = np.std(X_train, axis = 1).reshape(-1, 1) 
    #print("std_X: " + str(std_X.shape))

    # Normalise the data by subtracting mean_X and dividing element-wise by std_X

    # Training data
    X_train = X_train - mean_X
    X_train = X_train / std_X
    #print("X_train: " + str(X_train.shape)) # Shape intact. Normalisation seems to work

    # Validation data
    X_val = X_val - mean_X
    X_val = X_val / std_X
    #print("X_val: " + str(X_val.shape))

    # Test Data
    X_test = X_test - mean_X
    X_test = X_test / std_X
    #print("X_test: " + str(X_test.shape))

    labels = unpickle('cifar-10-batches-py/batches.meta')[ b'label_names']

    '''
    Below piece of code loads weights and biases in file defined by filename if load bool is True.
    '''

    W = None
    b = None
    if (load):
        savedWeights = np.load(filename)
        W = savedWeights['arr_0']
        b = savedWeights['arr_1']

    model = multiLinearClassifier(Y_train.shape[0], X_train.shape[0], W, b)

    #Test the classifier on a small subset of the data
    #P = model.evaluate_classifier(X_train[:, 0:100])
    #print("P: " + str(P.shape))

    '''
    J = model.compute_cost(X_train[:, 0:100], Y_train[:, 0:100], 0.01)
    print("J: " + str(J)) 

    accuracy = model.compute_accuracy(X_train[:, 0:100], y_train[0:100])
    print("Accuracy: " + str(accuracy))
    '''
    
    '''
    A little bit of code where both the numerical gradient and analytical gradient methods are tested to see if they are close enough.
    Analytical and numerical found to be accurate to 4 decimals.
    '''

    print(Y_train[:, :2])
    grad_W_test, grad_b_test = model.compute_gradients(X_train[:, :2], Y_train[:, :2], lamda=0)
    print("Grad W, analytical: " + str(np.linalg.norm(grad_W_test)))
    print("Grad b, analytical: " + str(np.linalg.norm(grad_b_test)))

    grad_W_test_num, grad_b_test_num = model.compute_gradients_num(X_train[:, :2], Y_train[:, :2], lamda=0)
    print("Grad W, numerical: " + str(np.linalg.norm(grad_W_test_num)))
    print("Grad b, numerical " + str(np.linalg.norm(grad_b_test_num)))

    print('Testing W-gradients...')
    compare_gradients(grad_W_test, grad_W_test_num)

    print('Testing b-gradients...')
    compare_gradients(grad_b_test, grad_b_test_num)
    

    '''
    lamdas = [0, 0, 0.1, 1]
    etas = [0.1, 0.001, 0.001, 0.001]

    W = np.copy(model.W)
    b = np.copy(model.b)

    for param in range(len(lamdas)):
        training_accuracies = []
        validation_accuracies = []
        test_accuracies = []

        model.W = np.copy(W) # To assure that all the training proceeds from the same starting point.
        model.b = np.copy(b)
        for i in range(10):
            
            print('Current run: ' + str(i))
            if(i == 0):
                plot = True
                
            else:
                plot = False

            if(i == 9):
                model.show_images(param, labels)
                plt.close()
            
            model.mini_batch_gradient_descent(X_train, Y_train, X_val, Y_val, eta = etas[param], lamda = lamdas[param], plot = plot)

            training_accuracies.append(model.compute_accuracy(X_train, y_train))
            validation_accuracies.append(model.compute_accuracy(X_val, y_val))
            test_accuracies.append(model.compute_accuracy(X_test, y_test))

        print('Settings: ' + 'eta = ' + str(etas[param]) + ' lamdas = ' + str(lamdas[param]))
        print('Training accuracy: ' + str(np.mean(training_accuracies)) + ' +- ' + str(np.std(training_accuracies)))
        print('Validation accuracy: ' + str(np.mean(validation_accuracies)) + ' +- ' + str(np.std(validation_accuracies)))
        print('Test accuracy: ' + str(np.mean(test_accuracies)) + ' +- ' + str(np.std(test_accuracies)))
    

    
    # Model parameters can be saved to .npz file if save bool is True
    if (save):
        W = np.copy(model.W)
        b = np.copy(model.b)
        np.savez(filename, W, b)
    '''