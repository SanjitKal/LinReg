import matplotlib.pyplot as pl
import numpy as np
import pandas as pd 

def gradientDescent(x, theta, y, alpha, iter, cost):
	for i in range(iter):
		y_hat = pd.Series(np.matmul(x,theta))
		residual = y_hat.subtract(y)
		theta[0] = theta[0] - (alpha/x.shape[0])*(np.sum(residual))
		delta_g = np.matmul(x.transpose(),residual)
		theta[1:] = theta[1:] - (alpha/x.shape[0])*(delta_g[1:])

		#Store cost during each iteration for plotting
		cost.append(np.sum(np.square(residual)))
	return theta

def normalEquation(x,y):
	pseudo_inv = np.linalg.inv(np.matmul(x.transpose(),x))
	transformation = np.matmul(pseudo_inv,x.transpose())
	theta = np.matmul(transformation,y)
	return theta

def predict(x, theta):
	theta_0 = theta[0]
	theta = theta[1:]
	#'x' should contain multiple columns for multivariate regression
	x = np.reshape(x,(7,1))
	return theta_0 + np.matmul(x,theta)

def main():
	#Store and prepare data
	cols = ['population','profit']
	df = pd.read_csv('ex1data1.txt', names = cols)
	x = df.iloc[:,0]
	x_norm = x.to_frame()
	#z-standardize dependednt variables
	x_norm = (x_norm-x_norm.mean())/(x_norm.std())
	#Add intercept column to design matrix
	x_norm.insert(0,'const',1)
	y = df.iloc[:,1]
	#Iinitialize random weights
	theta = pd.Series(np.random.uniform(-1,1,x_norm.shape[1]))
	#Train model using gradient descent
	cost = []
	theta_gd = gradientDescent(x_norm, theta, y, .01, 500, cost)
	# Train model using normal equation
	theta_ne = normalEquation(x_norm,y)
	# Plot results

	# Plot gradient descent regression results
	pl.subplot(221)
	pl.xlabel('z-Standardized Population')
	pl.ylabel('Profit')
	pl.title('Linear regression using Gradient Descent')
	pl.scatter(x=x_norm['population'],y=df['profit'])
	domain = np.arange(-2,5.0)
	pl.plot(domain,predict(domain,theta_gd))

	#Plot normal equation regression results
	pl.subplot(222)
	pl.xlabel('z-Standardized Population')
	pl.ylabel('Profit')
	pl.title('Linear regression using Normal Equation')
	pl.scatter(x=x_norm['population'],y=df['profit'])
	domain = np.arange(-2,5.0)
	pl.plot(domain, predict(domain,theta_ne))

	#Plot cost function
	pl.subplot(223)
	pl.xlabel('Iteration')
	pl.ylabel('Mean Squared Error')
	pl.title('MSE vs. Gradent Descent Iteration')
	domain = np.arange(500)
	pl.plot(domain,cost)
	axes = pl.gca()
	axes.set_xlim([0,len(domain)])
	axes.set_ylim([min(cost),max(cost)])

	pl.tight_layout()
	pl.show()

main()
