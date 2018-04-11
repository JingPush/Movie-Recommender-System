import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.utils import shuffle

sns.set()



def get_rating_matrix(df):
	r_df = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
	matrix = r_df.as_matrix()
	return matrix



def split_train_test(df, train_size, num_fold):
	#tes_index = np.random.choice(df.index, int(len(df.index)/k), replace=False)
	df = df.sample(frac=1)
	test_index = np.split(df.index, num_fold)
	test = []
	train = []
	for i in range(int(num_fold)):
		test.append(df.loc[test_index[i]])
		temp = df
		remain = temp.drop(temp.index[test_index[i]])
		train.append(remain.sample(n=train_size))
		#train.append(df.loc[~df.index.isin(test_index[i])].sample(n=train_size[i]))
	return train, test



def get_subset_value(df, new_df):
	temp = df
	temp.rating[new_df.index] = new_df.rating[new_df.index]
	temp.rating[~temp.index.isin(new_df.index)] = 0
	#r_df.ix[(r_df.userId.isin(new_df.userId)) & (r_df.movieId.isin(new_df.movieId)) == False, 'rating'] = 0
	return temp



def get_train_test_matrix(df, train_df, test_df):
	train_v = get_subset_value(df, train_df)
	print train_df.index
	print test_df.index
	print test_df
	print df.loc[test_df.index]
	#print len(df.rating[test_df.index])
	#print sum(df.rating[test_df.index] == test_df.rating[test_df.index])
	test_v = get_subset_value(df, test_df)
	train_v = get_subset_value(df, train_df)
	print "hhh", test_v.shape
	train = get_rating_matrix(train_v)
	test = get_rating_matrix(test_v)
	return train, test



	'''
	test = df.sample(frac=1/k)
	train = df.drop(test.index).sample(n=train_szie)
	return train, test
	'''



class CF():


	def __init__(self, matrix, k, num_iter, alpha, lamda):
		'''
		R: user-movie rating matrix
		k: number of latent features
		num_iter: number of iterations
		alpha: step size
		lamda: regularization parameter
		theta: user-latent feature parameter
		x: item-laten feature parameter
		'''
		self.R = matrix
		self.k = k
		self.num_users, self.num_items = matrix.shape
		self.num_iter = num_iter
		self.alpha = alpha
		self.lamda = lamda
		self.theta = None
		self.x = None

	'''
	def normalize_data(self):

		#subtract the mean of each user and fill in the missing values 0 with minus mean
		user_mean_rating = np.mean(self.R, axis=1)
		self.RR = self.R - user_mean_rating.reshape(-1, 1)
		self.mean = np.mean(self.R, axis=1)
	'''

	def train_CF(self):

		#self.normalize_data()

		# Initialize user and item latent feature matrix with Gaussian distribution
		self.theta = np.random.normal(scale=1./self.k, size=(self.num_users, self.k))
		self.x = np.random.normal(scale=1./self.k, size=(self.num_items, self.k))

		# add mean bias
		self.b = np.mean(self.R[np.where(self.R != 0)])

		#Initialize user and item bias term:
		self.b_u = np.zeros(self.num_users)
		self.b_m = np.zeros(self.num_items)

		self.train_gradient()



	def train_gradient(self):

		ctr = 0
		while ctr < self.num_iter:
			error = self.get_pred() - self.R
			error[self.R == 0] = 0

			
			#Update user-latent feature matrix and user bias
			for i in range(self.num_users):
				sum = np.zeros(self.k)
				for j in range(self.num_items):
					if self.R[i, j] != 0:
						sum += error[i, j] * self.x[j, :]
						#self.b_u[i] += self.alpha * (error[i, j]  - self.lamda * self.b_u[i])
				self.theta[i, :] -= self.alpha * (sum + self.lamda * self.theta[i, :])


			#Update item-latent feature matrix and item bias
			for j in range(self.num_items):
				sum = np.zeros(self.k)
				for i in range(self.num_users):
					if self.R[i, j] != 0:
						sum += error[i, j] * self.theta[i, :]
						#self.b_m[j] += self.alpha * (error[i, j] - self.lamda * self.b_m[j])
				self.theta[i, :] -= self.alpha * (sum + self.lamda * self.x[j, :])
		
			ctr += 1



	def get_pred(self):
		#mean = self.mean
		pred = np.dot(self.theta, self.x.T) + self.b
		return pred



def test_CF(train, test, k, num_iter, alpha, lamda):
	cf = CF(train, k, num_iter, alpha, lamda) 
	cf.train_CF()
	pred = cf.get_pred() 
	mse = mean_squared_error(pred[test!= 0], test)
	return mse



def cal_mse(pred_matrix, test_matrix):
	n = np.count_nonzero(test_matrix)
	error = np.square(np.linalg.norm(pred_matrix[test_matrix!=0]-test_matrix[test_matrix!=0]))/float(n)
	return error




def via_train_size(df, size, num_fold, k, num_iter, alpha, lamda):
	train_error = np.zeros((len(size), num_fold))
	test_error = np.zeros((len(size), num_fold))
	for i in range(len(size)):
		train_df, test_df = split_train_test(df, size[i], num_fold)
		for j in range(int(num_fold)):
			train, test = get_train_test_matrix(df, train_df[j], test_df[j])
			cf = CF(train, k, num_iter, alpha, lamda) 
			cf.train_CF()
			pred = cf.get_pred()
			#print train_df[j]
			#print train
			print train_df[j].shape
			print test_df[j].shape
			print train[train != 0].shape
			print test[test != 0].shape
			train_error[i, j] = mean_squared_error(pred[train != 0], train[train != 0])
			print "train error is", train_error[i, j], 'in folder', j, 'size', size[i]
			test_error[i, j] = mean_squared_error(pred[test != 0], test[test != 0])
			print "test error is", test_error[i, j], 'in folder', j, 'size', size[i]
	return error



def learning_curve(error, size, perc):
	e_mean = error.mean(axis=1)
	e_std = error.std(axis=1)
	for i in range(len(size)):
		e_std[i] = e_std[i] / np.sqrt(size[i])
	plt.xlabel('Train Percentage')
	plt.ylabel('MSE')
	plt.title('MSE vs. training size difference')
	plt.errorbar(perc, e_mean, color='k', yerr=e_std, marker='.')
	plt.legend(loc='center right')
	plt.show()






if __name__ == '__main__':

	#load data and select a subset as full dataset
	df = pd.read_csv('/users/jinggong/dropbox/2018 spring/cs578/Project/ml-20m/ratings.csv')
	df = pd.DataFrame(df, columns=['userId', 'movieId', 'rating', 'timestamp'])
	df = df[:2000]


	#Model via train_size difference by 10-folder incremental cv
	
	num_fold = 5
	k = 20
	num_iter = 50
	alpha = 0.1
	lamda = 0.05
	train_perc = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
	train_size = [int(item * df.shape[0]) for item in train_perc]
	'''
	train_df, test_df = split_train_test(df, train_size[1], num_fold)
	train = get_subset_value(df, train_df[0])
	test =  get_subset_value(df, test_df[0])
	df = get_rating_matrix(df)
	train = get_rating_matrix(train)
	test = get_rating_matrix(test)
	print df.shape
	print train.shape
	print test.shape
	#print get_train_test_matrix(df, train_df[0], test_df[0])
	'''
	error = via_train_size(df, train_size, num_fold, k, num_iter, alpha, lamda)
	#learning_curve(error, train_size, train_perc)
	





	'''
	R = get_rating_matrix(df[100:800])
	print R
	cf = CF(R, k=20, num_iter=50, alpha=0.1, lamda=0.05) 
	cf.train_CF()
	pred = cf.get_pred() 
	train_mse = mean_squared_error(pred[cf.R != 0], cf.R[cf.R != 0])
	print pred
	print train_mse
	'''











