# -*- coding: utf-8 -*-
# public static final String FileName = r"C:\Users\Shraddha\Desktop\SMLProject\mnist_data.mat";

import numpy as np
import scipy.stats
import scipy.io
import warnings
import matplotlib.cbook
import numpy as np
from pylab import *

# To suppress the mplDeprecation Warnings
warnings.filterwarnings("ignore", category = matplotlib.cbook.mplDeprecation)

# Class Definition of Naive_Bayes
class Naive_Bayes:
    def __init__(self, Num_of_Labels, Num_of_features):
        self.Num_of_Labels = np.array(Num_of_Labels)
        self.Num_of_features = np.array(Num_of_features)
        self.mean = np.zeros((Num_of_Labels, Num_of_features), dtype=np.float)
        self.var = np.zeros((Num_of_Labels, Num_of_features), dtype=np.float)
        self.pi = np.zeros(Num_of_Labels, dtype=np.float)

# Classifies the test data to 7 or 8
    def Predict(self, data):
        return self.__Predict_inner(data)

#Trains the model : input is Training data and Training labels
    def train(self, data, labels):
        N = data.shape[0] # the number of training data
        N_l = np.array([(labels == y).sum() for y in range(self.Num_of_Labels)], dtype=np.float) # count for each label

        # udpate mean of Gaussian
        for y in range(self.Num_of_Labels):
            sum = np.sum(data[n] if labels[n] == y else 0.0 for n in range(N))
            self.mean[y] = sum / N_l[y]

        # update variance of Gaussian
        for y in range(self.Num_of_Labels):
            sum = np.sum((data[n] - self.mean[y])**2 if labels[n] == y else 0.0 for n in range(N))
            self.var[y] = sum / N_l[y]

        # update prior of labels
        self.pi = N_l / N;

    def log_likelihood_Calc(self, data, labels):
        return self.__log_likelihood_inner_Calc(data, labels)

    def get_parameters(self):
        return ([self.mean, self.var], self.pi)

    def __Predict_inner(self, x):
        results = [self.__log_likelihood_inner_Calc(x, y) for y in range(self.Num_of_Labels)]
        return np.argmin(results)

    def __log_likelihood_inner_Calc(self, x, y):
        log_prior_y = -np.log(self.pi[y])
        log_posterior_x_given_y = -np.sum([self.guassian_pdf_log(x[d], self.mean[y][d], self.var[y][d]) for d in range(self.Num_of_features)])
        return log_prior_y + log_posterior_x_given_y

    def guassian_pdf_log(self, x, mean, var):
        esp = 1.0e-5
        if var < esp:
            return 0.0;
        # print('scipy.stats.norm(mean, var).logpdf(x) : ', scipy.stats.norm(mean, var).logpdf(x))
        return scipy.stats.norm(mean, var).logpdf(x)

def create_2D_images_horizontal(x, w, h):
    N = x.shape[0]
    for n in range(N):
        # print('n =',n,'N =',N)
        subplot_instance = subplot(1, N, n+1)
        subplot_instance.tick_params(labelleft='off',labelbottom='off')
        reshape_data = x[n].reshape(w, h)
        create_2D_image(reshape_data)

def create_2D_image(x):
    row, col = x.shape
    a = arange(col+1)
    b = arange(row+1)
    a, b = meshgrid(a, b)
    imshow(x)

def mnist_digit_recognition():
    # train_set, valid_set, test_set = load_mnist_dataset("mnist.pkl.gz")
    Numpyfile = scipy.io.loadmat(r'C:\Users\Shraddha\Desktop\SMLProject\mnist_data.mat')
    train_data = Numpyfile['trX']
    # print(len(train_data))
    train_labels = np.transpose(Numpyfile['trY'])
    # print(len(train_labels))
    test_data = Numpyfile['tsX']
    labels = np.transpose(Numpyfile['tsY'])
    Num_of_Labels = 2 # 7,8
    # Num_of_Labels = 10 # 1,2,3,4,5,6,7,9,0
    Num_of_features = 28*28
    mnist_model = Naive_Bayes(Num_of_Labels, Num_of_features)
    mnist_model.train(train_data,train_labels)

    [mean, var], pi = mnist_model.get_parameters()

    # visualization of learned means
    create_2D_images_horizontal(mean, w=28, h=28)
    show()

    limit = len(test_data)
    # limit = 50
    test_data, labels = test_data[:limit], labels[:limit]
    results = np.arange(limit, dtype=np.int)
    for n in range(limit):
        results[n] = mnist_model.Predict(test_data[n])
        # print ("%d : predicted %s, correct %s" % (n, results[n], labels[n]))
    print ("Naive Bayes accuracy: ", (results == labels).mean())

if __name__=="__main__":
    mnist_digit_recognition()
