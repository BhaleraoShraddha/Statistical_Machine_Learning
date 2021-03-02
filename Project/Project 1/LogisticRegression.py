import math
import numpy as np
import scipy.io

##The __Sig_Func function adjusts the cost function hypotheses to adjust the algorithm proportionally for worse estimations
def __Sig_Func(z):
	return 1 / (1 + np.exp(-z))

#Calculation of each instance of costfunction will be done using this function
#function takes inputs and returns outputs.
# To generate probabilities, logistic regression uses a function that gives outputs between 0 and 1 for all values of X.
def __hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return __Sig_Func(z)

#This function creates the gradient component for each Theta value
# by taking the parameters training data set, theta, cost function result and learning rate
def __Cost_Fun_Derv(X,Y,theta,j,m,alpha):
	SumErr = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = __hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		SumErr += error
	m = len(Y)
	constant = float(alpha)/float(m)
    # print(constant)
	J = constant * SumErr
	return J

#Calcuates the new theat values by substracting output of __Cost_Fun_Derv from previous theta value
def __Gradient_Algo(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
    # print(constant)
	for j in range(len(theta)):
		CFDerivative = __Cost_Fun_Derv(X,Y,theta,j,m,alpha)
		new_theta_val = theta[j] - CFDerivative
		new_theta.append(new_theta_val)
	return new_theta

#Manages the very high or low probability estimations and corresponsdingly changes the gradient of theta
def __Cost_Func_calc(X,Y,theta,m):
	sum_Of_Err = 0
	for i in range(m):
		xi = X[i]
		hi = __hypothesis(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sum_Of_Err += error
	const = -1/m
	J = const * sum_Of_Err
	# print ('cost is ', J )
	return J

#Calculate the prediction accuracy of algorithm
def __Accuracy_Calc(theta):
    Count = 0
    lenOfTestData = len(test_data)
    # lenOfTestData = 50 #For Debugging Purpose
    for i in range(lenOfTestData):
        prediction = np.round(__hypothesis(theta,test_data[i]))
        # print('%d : predicted %s, correct %s' % (i, prediction, labels[i]))
        answer = labels[i]
        if prediction == answer:
            Count += 1
    # print(lenOfTestData)
    # print(Count)
    Accuracy = float(Count) / float(lenOfTestData)
    print ('Logistic Regression Accuracy: ', Accuracy)


def LogisticReg(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in range(num_iters):
		new_theta = __Gradient_Algo(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			#here the cost function is used to present the final __hypothesis of the model in the same form for each gradient-step iteration
			__Cost_Func_calc(X,Y,theta,m)
	__Accuracy_Calc(theta)
    # for n in range(limit):
    #     results[n] = classify(test_data[n],theta)
    #     print ("%d : predicted %s, correct %s" % (n, results[n], labels[n]))
    # print ("recognition rate: ", (results == labels).mean())

if __name__ == "__main__":
    Numpyfile = scipy.io.loadmat(r'C:\Users\Shraddha\Desktop\SMLProject\mnist_data.mat')

# creating testing and training set
    X = Numpyfile['trX']
    # print(len(X))
    Y = np.transpose(Numpyfile['trY'])
    # print(len(Y))
    test_data = Numpyfile['tsX']
    labels = np.transpose(Numpyfile['tsY'])

# These are the initial guesses for theta as well as the learning rate of the algorithm
# Each iteration increases model accuracy but with diminishing returns,
    initial_theta = [0,0]
    alpha = 0.01
    iterations = 10000
    LogisticReg(X,Y,alpha,initial_theta,iterations)
