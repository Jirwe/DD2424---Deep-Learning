# Made by Marcus Jirwe for DD2424

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import pickle

def load_text(file):
    ''' Loads in the text for the RNN (The Goblet of Fire in this assignment) and creates a mapping from the character to its index and the other way around.
        The method to extract the unique characters sourced from: https://stackoverflow.com/questions/13902805/list-of-all-unique-characters-in-a-string

    Args:
        file, a string filename of the text to be read.

    Output:
        book_data, the text of file stored in string format.
        char_to_ind, a dictionary containing the mappings with a character as a key and its index (for its one-hot encoding) as the value
        ind_to_char, a dictionary with the inverse mapping from index to character.
        K, a scalar that denotes the size of our vocabulary, which will be used in other functions later. Determines the size of the one-hot representations.
    '''

    book_data = open(file, 'r', encoding='utf8').read()
    book_chars = list(set(book_data))

    K = len(book_chars)

    ind_to_char = {} 
    char_to_ind = {}
    
    for (ind, char) in enumerate(book_chars):
        ind_to_char[ind] = char
        char_to_ind[char] = ind

    return book_data, char_to_ind, ind_to_char, K
    

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

    for param in ana_grads: # Loops through the different parameters, like 'W'
        ana_norm = np.linalg.norm(ana_grads[param])
        num_norm = np.linalg.norm(num_grads[param])
        diff_norm = np.linalg.norm(ana_grads[param] - num_grads[param])

        rel_diff = diff_norm/(ana_norm + num_norm + np.finfo(np.float).eps) # Epsilon because of possible numerical stability issues.

        print('The gradient relative difference for ' + param + ': ' + str(rel_diff))

class RNN():
    ''' A RNN class containing all the methods for training and parameters, plus the associated dictionaries for going from characters to indices. '''

    def __init__(self, char_to_ind, ind_to_char, m = 100, K = 26, eta = 0.1, seq_length = 25, sig = 0.01):
        ''' Constructor for the RNN class. Takes in the mapping dictionaries and hyper-parameters.

        Args:
            char_to_ind, a dictionary containing the mappings from a character to the associated index.
            ind_to_char, a dictionary containing the mappings from an index to the associated character.
            m, an integer representing the dimensionality of the hidden state of the RNN.
            K, an integer representing the amount of characters in the mappings and thus the dimensionality of the one-hot representations.
            eta, a scalar defining the learning rate of the RNN.
            seq_length, an integer denoting the length of the input sequences used during training of the RNN.
            sig, the standard deviation to be used as the spread for the random initialisation of the parameters of the network.
        
        Output:
            self, a RNN object with the associated parameters, methods and functions necessary for training.
        '''

        self.char_to_ind = char_to_ind
        self.ind_to_char = ind_to_char
        self.m = m
        self.K = K
        self.eta = eta
        self.seq_length = seq_length

        # Initialises all the actual network parameters as specified in the assignment description.

        self.b = np.zeros((m, 1))
        self.c = np.zeros((K, 1))
        self.U = np.random.normal(0, sig, (m, K))
        self.W = np.random.normal(0, sig, (m, m))
        self.V = np.random.normal(0, sig, (K, m))

        # Initialises dictionaries containing the network parameters and the "momentum" of each parameter in m_dict.

        self.params_dict = {'W' : self.W, 'V' : self.V, 'U' : self.U, 'b' : self.b, 'c' : self.c}
        self.m_dict = {'W' : np.zeros_like(self.W), 'V' : np.zeros_like(self.V), 'U' : np.zeros_like(self.U), 'b' : np.zeros_like(self.b), 'c' : np.zeros_like(self.c)}

    def softmax(self, o):
        ''' Computes the softmax of an array and returns it.

        Args:
            o, an array of scalar values
        Output:
            p, a matrix of the same size with softmax:ed values, correpsonding to probabilities.
        '''

        p = np.exp(o - np.max(o, axis = 0)) / np.sum(np.exp(o - np.max(o, axis = 0)), axis = 0)

        return p

    def tanh(self, a):
        ''' Computes the tanh of an array and returns it.

        Args:
            a, an array of scalar values
        Output:
            h, an array of the same size with values that are tanh of the corresponding input.
        '''

        h = np.tanh(a)

        return h

    def calculate_forward_pass(self, h_prev, X):
        ''' Function that performs the calculations in equations (1)-(4) and returns all the outputs including the intermediary values.

        Args:
            hprev, a vector of representing the hidden state of the previous time step (h_(t-1) in assignment description). Dim (mx1)
            x, a matrix containing vectors of input text. Has dimension (dx1)

        Output:
            A, a matrix containing vectors of h and x transformed by W, U and b. Evaluated with equation (1).
            H, a matrix containing the following hidden states using the tanh function. Evaluated with equation (2).
            P, a matrix of probablities for each time step obtained from taking the softmax of "o". Evaluated with equation (4).
        '''

        tau = X.shape[1]

        A = np.zeros((self.m, tau))
        H = np.zeros((self.m, tau))
        P = np.zeros((self.K, tau))

        h = h_prev

        for t in range(tau):
            a = np.matmul(self.W, h) + np.matmul(self.U, X[:, [t]]) + self.b # Equation (1)
            h = self.tanh(a) # Equation (2)
            o = np.matmul(self.V, h) + self.c # Equation (3)
            p = self.softmax(o) # Equation (4)

            A[:, [t]] = a # Assign column t of A, H and P to be the recently calculated a, h & p.
            H[:, [t]] = h
            P[:, [t]] = p

        return A, H, P


    def synthesize_text(self, h, x, n):
        ''' Synthesize new text based on the input hidden state sequence as well as a dummy input.

        Args:
            h, a vector representing the previous hidden state sequence.
            x, a vector representing dummy input so that new states can be synthesized.
            n, an integer representing the length of the sequence to be generated (length in characters).

        Output:
            new_text, a string of n characters that denotes the newly synthesized sequence of characters.
        '''

        new_text = ''

        for t in range(n):
            _, h, p = self.calculate_forward_pass(h, x.reshape(-1, 1))

            # Sample an index of a corresponding character to be the next state vector. The probability distribution is given from the forward pass above.
            ind = np.random.choice(np.arange(self.K), p = p.flat)
            x = np.eye(self.K)[ind] # Extract the ind:th-row of the KxK identity matrix, being the one-hot enconding of the index.
            new_text += self.ind_to_char[ind] # Appends the corresponding character to the sampled index to the synthesized text.

        return new_text

    def compute_loss(self, X, Y, h_prev):
        ''' Compute the loss and cost according to equation (5)

        Args:
            X, a matrix of one-hot representations of the character-indices in the input. Dim (K x tau)
            Y, a matrix of one-hot representations of the character-indices in the labels. Dim (K x tau)
            h_prev, a previous (dummy) hidden state to begin the loss calculation.
        Output:
            l, a scalar denoting the cross-entropy loss for the RNN.
        '''

        tau = X.shape[1] # The amount if input vectors (one-hot) that we have.

        h = h_prev

        l = 0

        for t in range(tau): # Equation (5)
            _, h, p = self.calculate_forward_pass(h, X[:, t].reshape(-1, 1))

            l += -np.log(np.matmul(Y[:, t].T, p)) 

        return l.item() # To return as scalar and not 1x1 array


    def compute_gradients_num(self, X, Y, h_prev, h = 1e-5):
        ''' A numerical method calculation of the gradients. Adapted from the more precise matlab function. Uses the centered difference method.
        Default value for h taken as recommendation from assignment description.
        Essentially this function is an extension of the matlab function in python to work on the weights of both layers.
        Inspiration for flattening/vectorising parameters for gradient calculations from: https://towardsdatascience.com/coding-neural-network-gradient-checking-5222544ccc64
        With numpys 'flatten' function taking the part of the vectorising function in the link.

        Args:
            X, a matrix of data with dimension Kxtau.
            Y, a matrix of one-hot representations of labels. Dimension Kxtau. 
            h_prev, a vector which denotes the previous hidden state.
            h, a scalar which is the "width" of the interval used for the finite difference method.

        Output: 
            gradients, a dictionary containing the gradients for the parameters of the network.
        '''

        gradients = {'W' : np.zeros_like(self.W), 'V' : np.zeros_like(self.V), 'U' : np.zeros_like(self.U), 'b' : np.zeros_like(self.b), 'c' : np.zeros_like(self.c)}

        for param in self.params_dict: # The iterator over the different parameters (keys) in the parameter dictionary
            for i in range(len(self.params_dict[param].flatten())): # A one dimensional iterator over the elements of the current parameter.
                param_try = self.params_dict[param].flat[i]
                self.params_dict[param].flat[i] = param_try - h
                l1 = self.compute_loss(X, Y, h_prev) # Computed loss is just the same every time?
                self.params_dict[param].flat[i] = param_try + h
                l2 = self.compute_loss(X, Y, h_prev)
                self.params_dict[param].flat[i] = param_try # Restore old value after calculations.

                gradients[param].flat[i] = (l2 - l1) / (2*h)

        return gradients

    def compute_gradients(self, X, Y, h_prev):
        ''' Calculates the analytical gradients using equations from the notes of lecture 9.

        Args:
            X, a matrix of data with dimension Kxtau. The one-hot repr. of the input data.
            Y, a matrix of one-hot representations of labels. Dimension Kxtau. The one-hot repr. of the targets.
            h_prev, a vector that is the previous hidden state.

        Output: 
            gradients, a dictionary containing the gradients for the parameters of the network.
            h, the newest hidden state sequence. Would be used as the new h_prev
        '''

        tau = X.shape[1]

        gradients = {'W' : np.zeros_like(self.W), 'V' : np.zeros_like(self.V), 'U' : np.zeros_like(self.U), 'b' : np.zeros_like(self.b), 'c' : np.zeros_like(self.c)}

        A, H1, P = self.calculate_forward_pass(h_prev, X) # Performs the forward pass. The o values is not returned because they are not needed

        H0 = np.zeros((self.m, tau))
        H0[:, [0]] = h_prev # Sets the first element of H0 to be the initial dummy sequence.
        H0[:, 1:] = H1[:, :-1] # The same as H0 shifted by one hidden state sequence with the newest sequence at the end.

        G = (P.T - Y.T).T

        gradients['V'] = np.matmul(G, H1.T)
        gradients['c'] = np.sum(G, axis = -1, keepdims = True)

        g_h = np.zeros((tau, self.m))
        g_a = np.zeros((self.m, tau))

        g_h[-1] = np.matmul(G.T[-1], self.V)
        g_a[:, -1] = np.multiply(g_h[-1].T, (1 - np.square(self.tanh(A[:, -1]))))

        # Backward pass
        for t in reversed(range(tau-1)):
            g_h[t] = np.matmul(G.T[t], self.V) + np.matmul(g_a[:, t+1], self.W)
            g_a[:, t] = np.multiply(g_h[t].T, (1 - np.square(self.tanh(A[:, t]))))

        gradients['W'] = np.matmul(g_a, H0.T)
        gradients['U'] = np.matmul(g_a, X.T)
        gradients['b'] = np.sum(g_a, axis = -1, keepdims = True)
        # Essentially the same as the equation for the W-gradient, except the inner derivative of H0.T is replaced with a vector of ones.
        # This reduces the matrix multiplication/dot product to a sum over the entries of g_a 

        # clip the gradients to avoid exploding gradients.
        
        for gradient in gradients:
            gradients[gradient] = np.clip(gradients[gradient], -5, 5)
        
        h = H1[:, -1] # Picks out the most recent hidden state to return (to be used as the initial state for the next update iteration)

        return gradients, h.reshape(-1, 1)

    def ada_grad_descent(self, book_data, epochs = 10, eta = 0.1, seq_length = 25):
        '''Performs the learning of the model by using AdaGrad and traversing the book_data.

        Args:
            book_data, a string of text that the RNN will learn from.
            epochs, a scalar denoting how many times the book_data will be traversed through completely.
            eta, a scalar denoting the learning rate of the algorithm.
            seq_length, a scalar denoting how large of a text to read through at once. Analogous to the mini-batch size from the previous assignments.

        Output:
            None, since all the text is printed to the console and the the smooth loss plot is done with the program running, nothing is explicitly returned.
        '''

        N = len(book_data) # The amount of characters in book_data
        N_seq = int(np.ceil((N-1) / seq_length)) # The amount of sequences that can fit in the book data (-1 step, since the labels are one step ahead of the input)
        iter = 0
        smooth_loss = 0
        loss_list = []

        for epoch in range(epochs): # The outer epoch loop
            
            e = 0 # The position tracker in the book. Used to determine what mini_batch we are using, so to speak.
            h_prev = np.zeros((self.m, 1)) # Dummy hidden state to start off the epoch.

            print('Epoch: ' + str(epoch+1) + ' out of: ' + str(epochs))

            for seq in range(N_seq): # Inner iterator over the sequences.

                if seq == (N_seq-1): # Special case for last sequence so as to not reach out of bounds for the indices.
                    X_chars_ind = [self.char_to_ind[char] for char in book_data[e:N-2]]
                    Y_chars_ind = [self.char_to_ind[char] for char in book_data[e+1:N-1]]
                    e = N

                else:
                    X_chars_ind = [self.char_to_ind[char] for char in book_data[e:e+seq_length]]
                    Y_chars_ind = [self.char_to_ind[char] for char in book_data[e+1:e+seq_length+1]]
                    e += seq_length

                X = np.eye(self.K)[X_chars_ind].T
                Y = np.eye(self.K)[Y_chars_ind].T

                A, H1, P = self.calculate_forward_pass(h_prev, X)

                H0 = np.zeros((self.m, len(X_chars_ind)))

                H0[:, [0]] = h_prev
                H0[:, 1:] = H1[:, :-1]

                gradients, h_next = self.compute_gradients(X, Y, h_prev)

                loss = self.compute_loss(X, Y, h_prev)

                if smooth_loss == 0:
                    smooth_loss = loss

                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * loss

                h_prev = np.copy(h_next)

                # Update the momentum of all the gradient parameters:

                for param in self.m_dict:
                    self.m_dict[param] += np.square(gradients[param])

                # AdaGrad on all the parameters of the network.

                for param in self.params_dict:
                    self.params_dict[param] -= np.multiply((eta/np.sqrt(self.m_dict[param] + np.finfo(np.float).eps)), gradients[param])

                if iter % 100 == 0:
                    loss_list.append(smooth_loss)

                if iter % 1000 == 0:
                    print('Iter: ' + str(iter) + ', smooth loss: ' + str(smooth_loss))

                if iter % 10000 == 0:
                    synth_text = self.synthesize_text(h_prev, X[:, [0]], 200) # generate 200 characters of text.
                    print('Generated text from iteration: ' + str(iter) + '...')
                    print(synth_text)

                iter += 1 # Increase the iteration count.

        synth_text = self.synthesize_text(h_prev, X[:, [0]], 1000) # generate 1000 characters of text from the final model.
        print('Final generated text...')
        print(synth_text)

        plot = plt.plot(loss_list, label = 'Smooth loss')
        plt.title('Convergence of the smooth loss')
        plt.xlabel('Update iterations / 100')
        plt.ylabel('Smooth Loss')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    book_data, char_to_ind, ind_to_char, K = load_text('goblet_book.txt')

    model = RNN(char_to_ind, ind_to_char, K = K)

    # A short bit of test code where the gradients are tested against each other.
    '''
    X_char_ind = [char_to_ind[char] for char in book_data[0:2]]
    Y_char_ind = [char_to_ind[char] for char in book_data[1:3]]
    X = np.eye(K)[X_char_ind].T # So that the array has dimensions K x tau. One-hot encodings of the indices.
    print(X.shape)
    Y = np.eye(K)[Y_char_ind].T
    h_0 = np.zeros((100, 1)) # m = 100
    synth_text = model.synthesize_text(h_0, X[:, [0]], 200)
    print(synth_text)
    ana_grads, h = model.compute_gradients(X, Y, h_0)
    num_grads = model.compute_gradients_num(X, Y, h_0)
    compare_gradients(ana_grads, num_grads)
    '''

    # Have the untrained network generate 1000 characters of text.
    model.ada_grad_descent(book_data, epochs = 10, eta = 0.1, seq_length=25)
