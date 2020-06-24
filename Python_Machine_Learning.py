from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from numpy.random import seed
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



class Perceptron(object):
	def __init__(self,eta=0.01,n_iter=50,random_state=1):
		self.eta=eta
		self.n_iter=n_iter
		self.random_state=random_state

	def fit(self,X,y):
		rgen=np.random.RandomState(self.random_state)
		self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
		self.errors_=[]

		for _ in range(self.n_iter):
			errors=0
			for xi,target in zip(X,y):
				update=self.eta*(target-self.predict(xi))
				self.w_[1:]+=update*xi
				self.w_[0]+=update
				errors+=int(update!=0)
			self.errors_.append(errors)
		return self

	def net_input(self,X):
		return np.dot(X,self.w_[1:])+self.w_[0]

	def predict(self,X):
		return np.where(self.net_input(X)>=0.0,1,-1)


class AdalineGD(object):
	def __init__(self,eta=0.01,n_iter=50,random_state=1):
		self.eta=eta
		self.n_iter=n_iter
		self.random_state=random_state

	def fit(self,X,y):
		rgen=np.random.RandomState(self.random_state)
		self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
		self.cost_=[]

		for i in range(self.n_iter):
			net_input=self.net_input(X)
			output=self.activation(net_input)
			errors=(y-output)
			self.w_[1:]+=self.eta*X.T.dot(errors)
			self.w_[0]+=self.eta*errors.sum()
			cost=(errors**2).sum()/2.0
			self.cost_.append(cost)

		return self

	def net_input(self,X):
		return np.dot(X,self.w_[1:])+self.w_[0]

	def activation(self,X):
		return X

	def predict(self,X):
		return np.where(self.activation(self.net_input(X))>=0.0,1,-1)


class AdalineSGD(object):
	def __init__(self,eta=0.01,n_iter=10,shuffle=True,random_state=None):
		self.eta=eta
		self.n_iter=n_iter
		self.w_initialized=False
		self.shuffle=shuffle
		self.random_state=random_state

	def fit(self,X,y):
		self._initialize_weights(X.shape[1])
		self.cost_=[]
		for i in range(self.n_iter):
			if self.shuffle:
				X,y=self._shuffle(X,y)
			cost=[]
			for xi,target in zip(X,y):
				cost.append(self._update_weights(xi,target))
			avg_cost=sum(cost)/len(y)
			self.cost_.append(avg_cost)
		return self

	def partial_fit(self,X,y):
		if not self.w_initialized:
			self._initialize_weights(X.shape[1])
		if y.ravel().shape[0]>1:
			for xi,target in zip(X,y):
				self._update_weights(xi,target)
		else:
			self._update_weights(X,y)
		return self

	def _shuffle(self,X,y):
		r=self.rgen.permutation(len(y))
		return X[r],y[r]

	def _initialize_weights(self,m):
		self.rgen=np.random.RandomState(self.random_state)
		self.w_=self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
		self.w_initialized=True

	def _update_weights(self,xi,target):
		output=self.activation(self.net_input(xi))
		error=(target-output)
		self.w_[1:]+=self.eta*xi.dot(error)
		self.w_[0]+=self.eta*error
		cost=0.5*error**2
		return cost

	def net_input(self,X):
		return np.dot(X,self.w_[1:])+self.w_[0]

	def activation(self,X):
		return X

	def predict(self,X):
		return np.where(self.activation(self.net_input(X))>=0.0,1,-1)


class LogisticRegressionGD(object):
	def __init__(self,eta=0.05,n_iter=100,random_state=1):
		self.eta=eta
		self.n_iter=n_iter
		self.random_state=random_state

	def fit(self,X,y):
		rgen=np.random.RandomState(self.random_state)
		self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
		self.cost_=[]
		for i in range(self.n_iter):
			net_input=self.net_input(X)
			output=self.activation(net_input)
			errors=y-output
			self.w_[1:]+=self.eta*X.T.dot(errors)
			self.w_[0]+=self.eta*errors.sum()
			cost=-y.dot(np.log(output))-((1-y).dot(np.log(1-output)))
			self.cost_.append(cost)
		return self

	def net_input(self,X):
		return np.dot(X,self.w_[1:])+self.w_[0]

	def activation(self,z):
		return 1./(1.+np.exp(-np.clip(z,-250,250)))

	def predict(self,X):
		return np.where(self.net_input(X)>=0.0,1,0)



def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):
	markers=('s','x','o','^','v')
	colors=('red','blue','lightgreen','gray','cyan')
	cmap=ListedColormap(colors[:len(np.unique(y))])

	x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
	x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
	xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
	Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
	Z=Z.reshape(xx1.shape)
	plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
	plt.xlim(xx1.min(),xx1.max())
	plt.ylim(xx2.min(),xx2.max())

	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=colors[idx],marker=markers[idx],label=cl,edgecolor='black')

	if test_idx:
		X_test,y_test=X[test_idx,:],y[test_idx]
		plt.scatter(X_test[:,0],X_test[:,1],c='',edgecolor='black',alpha=1.0,linewidth=1,marker='o',s=100,label='test set')
