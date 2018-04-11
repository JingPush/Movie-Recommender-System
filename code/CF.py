import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds


def movie_genre_count(df):
    genre = df.genres
    my_list = []
    for line in genre:
        if '|' in line:
            new_line = list(line.split("|"))
            for item in new_line:
                my_list.append(item)
        else:
            my_list.append(line)
    count = len(Counter(my_list))
    return count



def get_user_item_matrix(df):
    n_users = df.userId.unique().shape[0]
    n_items = df.movieId.unique().shape[0]
    #print n_users
    #print n_items
    u_len = df.userId.max()
    m_len = df.movieId.max()
    matrix = np.zeros((u_len, m_len))
    df = df.values
    for i in range(u_len):
        for j in range(m_len):
            a = df[(df[:, 0] == (i+1)) & (df[:, 1] == (j+1)), 2]
            if len(a) > 0:
                matrix[i, j] = a[0]
    indice = np.any(matrix != 0, axis=0)
    matrix = matrix[:, indice]
    return matrix



def mean_normalization(matrix):
    m_mean = matrix.mean(axis=0)
    print 'hhh'
    new_matrix = matrix - m_mean[:, np.newaxis]
    print 'h'
    return new_matrix, m_mean


def train_GD(matrix, count):
    y = matrix
    n_user, n_item = matrix.shape
    x = np.random.normal(size=(count, n_item))
    #x = np.full((count, n_item), 0.01)
    theta = np.random.normal(size=(count, n_user))
    #theta = np.full((count, n_user), 0.01)
    iter = 0
    tol_u = 1
    tol_i = 1
    alpha = 0.05
    lamb = 0.01
    while ((iter < 50) & (tol_u > 1e-6) & (tol_i > 1e-6)):
        old_theta = theta
        old_x = x
        for j in range(n_user):
            for i in range(n_item):
                indice_j = (y[:, i] != 0)
                indice_i = (y[j, :] != 0)
                xx = x[:, i].reshape((count, 1))
                tt = theta[:, j].reshape((count, 1))
                yy_j = y[indice_j, i].reshape((1, y[indice_j, i].shape[0]))
                yy_i = y[j, indice_i].reshape((1, y[j, indice_i].shape[0]))
                gradient_i = np.dot(theta[:, indice_j], (np.dot(theta[:, indice_j].T, xx)
                                                         - yy_j.T)) + lamb * xx
                print gradient_i[:, 0]
                x[:, i] -= alpha * gradient_i[:, 0]
                gradient_j = np.dot(x[:, indice_i], (np.dot(x[:, indice_i].T, tt)
                                                     - yy_i.T)) + lamb * tt
                print gradient_j[:, 0]
                theta[:, j] -= alpha * gradient_j[:, 0]
        iter += 1
        print iter
        tol_u = np.linalg.norm(old_theta - theta)
        tol_i = np.linalg.norm(old_x - x)
    return theta, x



def main():
    df_movie = pd.read_csv('/users/jinggong/dropbox/2018 spring/cs578/Project/ml-20m/movies.csv')
    df_rating = pd.read_csv('/users/jinggong/dropbox/2018 spring/cs578/Project/ml-20m/ratings.csv')
    count = movie_genre_count(df_movie)
    print count
    matrix = get_user_item_matrix(df_rating[:2000])
    print matrix
    #new_matrix, m_mean = mean_normalization(matrix)
    #print new_matrix
    theta, x = train_GD(matrix, count)
    print theta
    print x



main()