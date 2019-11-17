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

#     def gradesc_dynamic(self,f=10,lam=0.3, batch = 50, lrate=0.01, maxiter=10, stopping_deriv=0.01):
#         return

    def gradesc_dynamic(self,f=10,lam=0.3, batch=50, lrate=0.01, maxiter=10, stopping_deriv=0.01, bin_num=30, gamma=0.03, power=0.25):
        # random assign value to matrix p and q
        user_latent = np.random.randn(f,self._u)
        user_latent = pd.DataFrame(user_latent)
        user_latent.columns = self._data['userId'].unique().tolist()

        item_latent = np.random.randn(f,self._i)
        item_latent = pd.DataFrame(item_latent)
        item_latent.columns = self._data['movieId'].unique().tolist()

        user_mean = self._data.groupby('userId').mean()['rating']
        self._user_bias = pd.DataFrame(user_mean - total_mean)
        

        ########## bias of item with respect to time #########

        # distribute to bins
        bin_width = (self._data['timestamp'].max()-self._data['timestamp'].min())//bin_num
        bin_loc = (self._data['timestamp']-self._data['timestamp'].min())//bin_width
        # adding info to origin dataset
        self._data['bin_loc'] = bin_loc
        item_bin_bias = self._data.groupby(['movieId','bin_loc']).mean()['rating']
        # bias: b_i
        total_mean = np.mean(data['rating'])
        item_mean = self._data.groupby('movieId').mean()['rating']
        item_bias = pd.DataFrame(item_mean - total_mean)
        self._item_bias = pd.DataFrame(item_mean - total_mean)
        # item dynamic bias: b_i_Bin(t)+b_i
        for item_id in item_bias.index:
          item_bin_bias[item_bin_bias.index.get_level_values(0)==item_id] += self._item_bias['rating'][item_id]
        self._item_bin_bias = item_bin_bias

        ########### bias of user with respect to time ###########
        ######################## by spline ###########################

        # bias: b_u
        total_mean = np.mean(data['rating'])
        user_mean = self._data.groupby('userId').mean()['rating']
        user_bias = pd.DataFrame(user_mean - total_mean)
        self._user_bias = pd.DataFrame(user_mean - total_mean)
        # Prep for splines
        user_rating_num = self._data.groupby(['userId','timestamp']).count()
        N_vec = []
        for user_id in data['userId'].unique():  
          N_vec.append(user_rating_num[user_rating_num.index.get_level_values(0)==user_id]['movieId'].sum())

        K_vec = np.round(np.power(N_vec,power))

        data_sort = self._data[['userId','timestamp','rating']].sort_values(by=['userId','timestamp'])
        new_index = []
        for user_id in data['userId'].unique():
          new_index.extend(list(range(1,N_vec[user_id-1]+1)))
        data_sort['old_index'] = data_sort.index
        data_sort.index = new_index

        step_vec = np.floor(N_vec/K_vec)
        user_time_bias = pd.Series([])

        for user_id in data_sort['userId'].unique():
         A = np.linspace(1,K_vec[user_id-1],int(K_vec[user_id-1]))*step_vec[user_id-1]
         A = A.astype(int).tolist()
         B = [1]
         B.extend(A)

         control_time = data_sort[data_sort['userId']==user_id]['timestamp'][B]
         control_time_bias = data_sort[data_sort['userId']==user_id]['rating'][B] - total_mean

         denom = 0
         numer = 0
         for i in range(len(B)):
           time_dev_abs = np.abs(data_sort[data_sort['userId']==user_id]['timestamp'] - control_time.iloc[i])
           denom = denom + np.exp(-gamma*time_dev_abs)
           numer = numer + np.exp(-gamma*time_dev_abs)*control_time_bias.iloc[i]
         spline_frac = numer/denom
         time_bias = spline_frac.fillna(0) + user_bias.iloc[user_id-1][0]
         user_time_bias = user_time_bias.append(time_bias)

        data_sort['user_time_bias'] = user_time_bias.tolist()
        data_sort.index = data_sort['old_index'].tolist()
        self._user_time_bias = data_sort.sort_index()[['userId','timestamp','user_time_bias']]

        sample_index = [index for index in self._train_data.index]

        for m in range(maxiter):
            random.sample(sample_index, batch)
            # loop through each training case and perform update
            for index in sample_index:
                u = self._train_data.loc[index]['userId']
                u_time = self._train_data.loc[index]['timestamp']
                i = self._train_data.loc[index]['movieId']
                i_bin = self._train_data.loc[index]['timestamp']//bin_width
                bias_u = self._user_time_bias[(self._user_time_bias['userId']==u)&(self._user_time_bias['timestamp']==u_time)]
                bias_i = self._item_bin_bias[i,i_bin]
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
    
    def calcSimMatrix(self):
        # similarity bewteen movie i1 and i2
        matSim = np.zeros((self._i, self._i))
        for i1 in range(self._i):
            for i2 in range(i1+1, self._):
                currentSim = pairwise_distances(self._item_latent[:,i1], self._item_latent[:,i2], metric='cosine')
                matSim[i1, i2] = currentSim
                matSim[i2, i1] = currentSim
        self._sim = matSim

    def KNN(self,k=1,test_point=None):
        if k is not None:
            self._k = k
        # compute distance from test point to all train point
        all_distance = [pairwise_distances(train_point, test_point, metric='cosine') for train_point in self._user_latent]

        # get nearest k neighbors' index
        k_neighbors_class = np.argsort(all_distance)[:self._k]
        # get nearest k neighbors most frequent label
        # DON'T KNOW WHAT IS THE RETURN
        return #Counter(self._y_train[k_neighbors_class]).most_common()[0][0]

    def predict(self,train_data = None, test_data = None):

        if train_data is not None and test_data is not None:
            self._train_data = train_data
            self._test_data = test_data

        train_RMSE = []
        test_RMSE = []
        est_rating = np.dot(self._item_latent.T, self._user_latent)
        est_rating = pd.DataFrame(est_rating)
        est_rating.index = self._data['movieId'].unique().tolist()
        est_rating.columns = self._data['userId'].unique().tolist()

        # add linear regression???

        train_RMSE_cur = RMSE(self._train_data, est_rating)
        train_RMSE.append(train_RMSE_cur)
        print("training RMSE:", train_RMSE_cur)
        test_RMSE_cur = RMSE(self._test_data, est_rating)
        test_RMSE.append(test_RMSE_cur)
        print("test RMSE:", test_RMSE_cur)

        return [train_RMSE, test_RMSE]
