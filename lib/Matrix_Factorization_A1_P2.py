import random
import pandas as pd
import numpy as np
#Define a function to calculate RMSE

def RMSE(rating, est_rating):
    sqr_error = []
    for r in range(rating.shape[0]):
        u = rating.iloc[r]['userId']
        i = rating.iloc[r]['movieId']
        r_ui = rating.iloc[r]['rating']
        est_r_ui = est_rating.loc[i,u]
        sqr_error.append((r_ui-est_r_ui)**2)
    return np.sqrt(np.mean(sqr_error))


# Stochastic Gradient Descent
# a function returns a list containing factorized matrices p and q, training and testing RMSEs.
def gradesc(f=10,U=None,I=None, lam=0.3, lrate=0.01, maxiter=10, stopping_deriv=0.01, data=None, train_data=None, test_data=None):
    # random assign value to matrix p and q
    p = np.random.uniform(-1, 1, (f, U))
    p = pd.DataFrame(p)
    p.columns = [_ + 1 for _ in range(U)]
    q = np.random.uniform(-1, 1, (f, I))
    q = pd.DataFrame(q)
    q.columns = data['movieId'].unique().tolist()
    train_RMSE = []
    test_RMSE = []
    for m in range(1, maxiter + 1):
        sample = train_data.sample(train_data.shape[0])
        # loop through each training case and perform update
        for j in sample.index:
            u = sample.loc[j]['userId']
            i = sample.loc[j]['movieId']
            r_ui = sample.loc[j]['rating']
            e_ui = r_ui - np.dot(q[[i]].T, p[[u]])
            grad_q = np.array(e_ui * p[[u]]) - np.array(lam * q[[i]])

            if (all(grad > stopping_deriv for grad in grad_q)):
                q[[i]] = q[[i]] + lrate * grad_q

            grad_p = np.array(e_ui * q[[i]]) - np.array(lam * p[[u]])
            if (all(grad > stopping_deriv for grad in grad_p)):
                p[[u]] = p[[u]] + lrate * grad_p
        # print the values of training and testing RMSE
        if (m % 10 == 0):
            print('epoch:', m)
            est_rating = np.dot(q.T, p)
            est_rating = pd.DataFrame(est_rating)
            est_rating.index = data['movieId'].unique().tolist()
            est_rating.columns = [_ + 1 for _ in range(U)]
            train_RMSE_cur = RMSE(train_data, est_rating)
            train_RMSE.append(train_RMSE_cur)
            print("training RMSE:", train_RMSE_cur)
            test_RMSE_cur = RMSE(test_data, est_rating)
            test_RMSE.append(test_RMSE_cur)
            print("test RMSE:", test_RMSE_cur)

    return [p, q,  train_RMSE, test_RMSE]
