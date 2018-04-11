import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


def get_rating_df(df):
    rating_df = pd.DataFrame(df, columns=['userId', 'movieId', 'rating', 'timestamp'])
    r_df = rating_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return r_df


def normalize_data(r_df):
    r_matrix = r_df.as_matrix()
    user_mean_rating = np.mean(r_matrix, axis=1)
    utility = r_matrix - user_mean_rating.reshape(-1, 1)
    return utility, user_mean_rating

def train_svds(utility, k):
    u, sigma, vt = svds(utility, k)
    sigma = np.diag(sigma)
    return u, sigma, vt

def pred_svds(U, sigma, Vt, user_mean_rating):
    predict_rating = np.dot(np.dot(U, sigma), Vt) + user_mean_rating.reshape(-1, 1)
    return predict_rating


def test_svds(u, sigma, vt, user_mean_rating, utility):
    predict_rating = np.dot(np.dot(u, sigma), vt) + user_mean_rating.reshape(-1, 1)
    error = sqrt(mean_squared_error(utility, predict_rating))
    return error



def main():
    df_rating = pd.read_csv('/users/jinggong/dropbox/2018 spring/cs578/Project/ml-20m/ratings.csv')
    r_df_train = get_rating_df(df_rating[:50000])
    utility, user_mean_rating = normalize_data(r_df_train)
    ''''
    u, sigma, vt = train_svds(utility, k=50)
    pred_rating = pred_svds(u, sigma, vt, user_mean_rating)
    print pred_rating
    error = test_svds(u, sigma, vt, user_mean_rating, utility)
    print error
    '''
    #choose k based on train error(should implement cross-validation later!)
    k_list = [20, 30, 40, 50, 60, 70, 80, 90]
    RMSE = np.zeros((len(k_list), 1))
    for i in range(len(k_list)):
        print i
        k = k_list[i]
        u, sigma, vt = train_svds(utility, k)
        RMSE[i] = test_svds(u, sigma, vt, user_mean_rating, utility)
        print RMSE[i]
    plt.plot(k_list, RMSE, 'r')
    plt.xlabel("k - dimension of latent factors")
    plt.ylabel("RMSE")
    plt.title("RMSE of train error via number of latent factors")
    plt.show()
    '''
    size = [0.1, ]
    RMSE = np.zeros((len(k_list), 1))
    for i in range(len(k_list)):
        print i
        k = k_list[i]
        u, sigma, vt = train_svds(utility, k)
        RMSE[i] = test_svds(u, sigma, vt, user_mean_rating, utility)
        print RMSE[i]
    plt.plot(k_list, RMSE, 'r')
    plt.xlabel("k - dimension of latent factors")
    plt.ylabel("RMSE")
    plt.title("RMSE of train error via number of latent factors")
    plt.show()
    '''
    return


main()