import random
import pandas as pd
import numpy as np
from Evaluation import RMSE
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances

#Define a function to calculate RMSE
class Matrix_Factorization:
    def __init__(self,data = None, train_data=None,test_data=None):
        if data is not None and train_data is not None and test_data is not None:
            self._data = data
            self._train_data = train_data
            self._test_data = test_data
            self._u = len(self._data['userId'].unique())
            self._i = len(self._data['movieId'].unique())
            

# Stochastic Gradient Descent
# a function returns a list containing factorized matrices p and q, training and testing RMSEs.
    def gradesc(self,f=10,lam=0.3, batch=50, lrate=0.01, maxiter=10, stopping_deriv=0.01):
        # random assign value to matrix p and q
        user_latent = np.random.randn(f,self._u)
        user_latent = pd.DataFrame(user_latent)
        user_latent.columns = self._data['userId'].unique().tolist()

        item_latent = np.random.randn(f,self._i)
        item_latent = pd.DataFrame(item_latent)
        item_latent.columns = self._data['movieId'].unique().tolist()


        sample_index = [index for index in self._train_data.index]

        for m in range(maxiter):
            sample_batch = random.sample(sample_index, batch)
            # loop through each training case and perform update
            for index in sample_batch:
                u = self._train_data.loc[index]['userId']
                i = self._train_data.loc[index]['movieId']
                r_ui = self._train_data.loc[index]['rating']
                e_ui = r_ui - np.dot(item_latent[[i]].T, user_latent[[u]])
                grad_q = np.array(e_ui * user_latent[[u]]) - np.array(lam * item_latent[[i]])

                if (all(grad > stopping_deriv for grad in grad_q)):
                    item_latent[[i]] = item_latent[[i]] + lrate * grad_q

                grad_p = np.array(e_ui * item_latent[[i]]) - np.array(lam * user_latent[[u]])
                if (all(grad > stopping_deriv for grad in grad_p)):
                    user_latent[[u]] = user_latent[[u]] + lrate * grad_p

            self._user_latent = user_latent
            self._item_latent = item_latent
        print(m, 'user_latent', user_latent)



    def gradesc_bias(self,f=10,lam=0.3, batch = 50, lrate=0.01, maxiter=10, stopping_deriv=0.01):
        # random assign value to matrix p and q
        user_latent = np.random.randn(f,self._u)
        user_latent = pd.DataFrame(user_latent)
        user_latent.columns = self._data['userId'].unique().tolist()

        item_latent = np.random.randn(f,self._i)
        item_latent = pd.DataFrame(item_latent)
        item_latent.columns = self._data['movieId'].unique().tolist()

        user_mean = self._data.groupby('userId').mean()['rating']
        item_mean = self._data.groupby('movieId').mean()['rating']
        total_mean = np.mean(self._data['rating'])
        self._user_bias = pd.DataFrame(user_mean - total_mean)
        self._item_bias = pd.DataFrame(item_mean - total_mean)

        sample_index = [index for index in self._train_data.index]

        for m in range(maxiter):
            random.sample(sample_index, batch)
            # loop through each training case and perform update
            for index in sample_index:
                u = self._train_data.loc[index]['userId']
                i = self._train_data.loc[index]['movieId']
                bias_u = self._user_bias.loc[u]['rating']
                bias_i = self._item_bias.loc[i]['rating']
                r_ui = self._train_data.loc[index]['rating']
                e_ui = r_ui - total_mean-bias_u -bias_i- np.dot(item_latent[[i]].T, user_latent[[u]])

                grad_user_latent = np.array(e_ui * user_latent[[u]]) - np.array(lam * item_latent[[i]])
                if (all(grad > stopping_deriv for grad in grad_user_latent)):
                    item_latent[[i]] = item_latent[[i]] + lrate * grad_user_latent

                grad_item_latent = np.array(e_ui * item_latent[[i]]) - np.array(lam * user_latent[[u]])
                if (all(grad > stopping_deriv for grad in grad_item_latent)):
                    user_latent[[u]] = user_latent[[u]] + lrate * grad_item_latent

                grad_user_bias = e_ui - lam*bias_u
                if (all(grad > stopping_deriv for grad in grad_user_bias)):
                    self._user_bias.loc[u]['rating'] = bias_u + lrate * grad_user_bias

                grad_item_bias = e_ui - lam * bias_i
                if (all(grad > stopping_deriv for grad in grad_item_bias)):
                    self._item_bias.loc[i]['rating'] = bias_u + lrate * grad_item_bias

            self._user_latent = user_latent
            self._item_latent = item_latent

            print(m, 'user_latent', user_latent)

    def gradesc_dynamic(self,f=10,lam=0.3, batch = 50, lrate=0.01, maxiter=10, stopping_deriv=0.01):
        return


                
    def KNN(self,k=5,test_point=None):
        self._sim = pairwise_distances(self._item_latent.T,metric='cosine')
        self._sim = pd.DataFrame(self._sim)
        self._sim.index = self._item_latent.columns
        self._sim.columns = self._item_latent.columns
        
        r_ij = np.dot(self._item_latent.T, self._user_latent)
        
        if k is not None:
            self._k = k
        # compute distance from test point to all train point
        #all_distance = [pairwise_distances(train_point, test_point, metric='cosine') for train_point in self._user_latent]

        # get nearest k neighbors' index
        #k_neighbors_class = np.argsort(all_distance)[:self._k]
        knn_r_ij = []
        tmp = test_point['movieId'].unique()
        for i in range(len(tmp)):
            k_neighbors_class = np.argsort(self._sim[tmp[i]])[1:1+self._k]
            knn_r_ij.append(np.mean(r_ij[k_neighbors_class,:],axis=0))
        knn_r_ij = pd.DataFrame(knn_r_ij)
        knn_r_ij.index = self._item_latent.columns
        knn_r_ij.columns = self._user_latent.columns
        
        self._est_rating = knn_r_ij
        # get nearest k neighbors most frequent label
        # DON'T KNOW WHAT IS THE RETURN
        #return #Counter(self._y_train[k_neighbors_class]).most_common()[0][0]
        return knn_r_ij

    def predict(self,train_data = None, test_data = None):

        if train_data is not None and test_data is not None:
            self._train_data = train_data
            self._test_data = test_data

        train_RMSE = []
        test_RMSE = []
        #est_rating = np.dot(self._item_latent.T, self._user_latent)
        #est_rating = pd.DataFrame(est_rating)
        #est_rating.index = self._data['movieId'].unique().tolist()
        #est_rating.columns = self._data['userId'].unique().tolist()
        est_rating = self._est_rating
        # add linear regression???

        train_RMSE_cur = RMSE(self._train_data, est_rating)
        train_RMSE.append(train_RMSE_cur)
        print("training RMSE:", train_RMSE_cur)
        test_RMSE_cur = RMSE(self._test_data, est_rating)
        test_RMSE.append(test_RMSE_cur)
        print("test RMSE:", test_RMSE_cur)

        return [train_RMSE, test_RMSE]
