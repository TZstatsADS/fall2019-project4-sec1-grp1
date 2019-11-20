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
    def gradesc(self,f=10,lam=0.3, lrate=0.01, epoch=10, stopping_deriv=0.01):
        # random assign value to matrix p and q
        user_latent = np.random.randn(f,self._u)*0.01
        #user_latent = pd.DataFrame(user_latent)
        #user_latent.columns = self._data['userId'].unique().tolist()

        item_latent = np.random.randn(f,self._i)*0.01
        #item_latent = pd.DataFrame(item_latent)
        #item_latent.columns = self._data['movieId'].unique().tolist()
        
        tmp1 = [i for i in range(self._i)]
        tmp2 = self._data['movieId'].unique()
        self._movie_dict = dict(zip(tmp2,tmp1))
        
        train_data = np.array(self._train_data)
        
        sample_index = [index for index in range(train_data.shape[0])]

        for e in range(epoch):
            #sample_batch = random.sample(sample_index, batch)
            random.shuffle(sample_index)
            # loop through each training case and perform update
            for index in sample_index:
                u = int(train_data[index,0])
                i = int(train_data[index,1])
                r_ui = train_data[index,2]
                e_ui = r_ui - np.dot(item_latent[:,self._movie_dict[i]].T, user_latent[:,u-1])
                
                grad_q = e_ui * user_latent[:,u-1] - lam * item_latent[:,self._movie_dict[i]]
                if (all(np.abs(grad) > stopping_deriv for grad in grad_q)):
                    item_latent[:,self._movie_dict[i]] = item_latent[:,self._movie_dict[i]] + lrate * grad_q

                grad_p = e_ui * item_latent[:,self._movie_dict[i]] - lam * user_latent[:,u-1]
                if (all(np.abs(grad) > stopping_deriv for grad in grad_p)):
                    user_latent[:,u-1] = user_latent[:,u-1] + lrate * grad_p
 
            self._user_latent = user_latent
            self._item_latent = item_latent
            self._model = 'basic'



    def gradesc_bias(self,f=10,lam=0.3, lrate=0.01, epoch=10, stopping_deriv=0.01):
        # random assign value to matrix p and q
        user_latent = np.random.randn(f,self._u)*0.01
        #user_latent = pd.DataFrame(user_latent)
        #user_latent.columns = self._data['userId'].unique().tolist()

        item_latent = np.random.randn(f,self._i)*0.01
        #item_latent = pd.DataFrame(item_latent)
        #item_latent.columns = self._data['movieId'].unique().tolist()
        
        tmp1 = [i for i in range(self._i)]
        tmp2 = self._data['movieId'].unique()
        self._movie_dict = dict(zip(tmp2,tmp1))
        
        train_data = np.array(self._train_data)
        
        user_mean = self._data.groupby('userId').mean()['rating']
        user_mean = np.array(user_mean)
        
        item_mean = self._data.groupby('movieId').mean()['rating']
        tmp_item_index = item_mean.index.tolist()
        tmp_movie_dict = dict(zip(tmp_item_index,[i for i in range(self._i)]))
        item_mean = np.array(item_mean)
        
        total_mean = np.mean(self._data['rating'])
        self._total_mean = total_mean
        self._user_bias = user_mean - total_mean
        self._item_bias = item_mean - total_mean

        sample_index = [index for index in range(train_data.shape[0])]

        for e in range(epoch):
            random.shuffle(sample_index)
            # loop through each training case and perform update
            for index in sample_index:
                u = int(train_data[index,0])
                i = int(train_data[index,1])
                r_ui = train_data[index,2]
                bias_u = self._user_bias[u-1]
                bias_i = self._item_bias[tmp_movie_dict[i]]
                e_ui = r_ui - total_mean - bias_u - bias_i- np.dot(item_latent[:,self._movie_dict[i]].T, user_latent[:,u-1])

                grad_user_latent = e_ui * user_latent[:,u-1] - lam * item_latent[:,self._movie_dict[i]]
                if (all(np.abs(grad) > stopping_deriv for grad in grad_user_latent)):
                    item_latent[:,self._movie_dict[i]] = item_latent[:,self._movie_dict[i]] + lrate * grad_user_latent

                grad_item_latent = e_ui * item_latent[:,self._movie_dict[i]] - lam * user_latent[:,u-1]
                if (all(np.abs(grad) > stopping_deriv for grad in grad_item_latent)):
                    user_latent[:,u-1] = user_latent[:,u-1] + lrate * grad_item_latent

                grad_user_bias = e_ui - lam * bias_u
                if (np.abs(grad_user_bias) > stopping_deriv):
                    self._user_bias[u-1] = bias_u + lrate * grad_user_bias

                grad_item_bias = e_ui - lam * bias_i
                if (np.abs(grad_item_bias) > stopping_deriv):
                    self._item_bias[tmp_movie_dict[i]] = bias_i + lrate * grad_item_bias

                
            self._user_latent = user_latent
            self._item_latent = item_latent
            
            self._model = 'bias'

    def gradesc_dynamic(self,f=10,lam=0.3, lrate=0.01, epoch=10, stopping_deriv=0.01,bin_num = 30, gamma=0.3, power=0.25):
        # random assign value to matrix p and q
        user_latent = np.random.randn(f,self._u)*0.01
        item_latent = np.random.randn(f,self._i)*0.01

        ########## bias of item with respect to time #########

        # distribute to bins
        bin_width = (self._data['timestamp'].max()-self._data['timestamp'].min())//bin_num
        bin_loc = (self._data['timestamp']-self._data['timestamp'].min())//bin_width
        # adding info to origin dataset
        self._data['bin_loc'] = bin_loc
        item_bin_bias = self._data.groupby(['movieId','bin_loc']).mean()['rating']
        # bias: b_i
        total_mean = np.mean(self._data['rating'])
        item_mean = self._data.groupby('movieId').mean()['rating']
        item_bias = pd.DataFrame(item_mean - total_mean)
        for i in self._data['movieId'].unique():
          item_bin_bias[i] -= item_mean[i] 
        self._item_bias = pd.DataFrame(item_mean - total_mean)
        self._item_bin_bias = item_bin_bias - total_mean

        ########### bias of user with respect to time ###########
        ######################## by spline ###########################

        # bias: b_u
        total_mean = np.mean(self._data['rating'])
        user_mean = self._data.groupby('userId').mean()['rating']
        user_bias = pd.DataFrame(user_mean - total_mean)
        self._user_bias = pd.DataFrame(user_mean - total_mean)
        # Prep for splines
        user_rating_num = self._data.groupby(['userId','timestamp']).count()
        N_vec = []
        for user_id in self._data['userId'].unique():  
          N_vec.append(user_rating_num[user_rating_num.index.get_level_values(0)==user_id]['movieId'].sum())

        K_vec = np.round(np.power(N_vec,power))

        data_sort = self._data[['userId','timestamp','rating']].sort_values(by=['userId','timestamp'])
        new_index = []
        for user_id in self._data['userId'].unique():
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
         time_bias = spline_frac.fillna(0) 
         user_time_bias = user_time_bias.append(time_bias)

        data_sort['user_time_bias'] = user_time_bias.tolist()
        data_sort.index = data_sort['old_index'].tolist()
        self._user_time_bias = data_sort.sort_index()[['userId','timestamp','user_time_bias']]
        
        
        tmp1 = [i for i in range(self._i)]
        tmp2 = self._data['movieId'].unique()
        self._movie_dict = dict(zip(tmp2,tmp1))
        
        train_data = np.array(self._train_data)
        
        user_mean = self._data.groupby('userId').mean()['rating']
        user_mean = np.array(user_mean)
        
        item_mean = self._data.groupby('movieId').mean()['rating']
        tmp_item_index = item_mean.index.tolist()
        tmp_movie_dict = dict(zip(tmp_item_index,[i for i in range(self._i)]))
        item_mean = np.array(item_mean)
        
        total_mean = np.mean(self._data['rating'])
        self._total_mean = total_mean
        self._user_bias = user_mean - total_mean
        self._item_bias = item_mean - total_mean

        sample_index = [index for index in range(train_data.shape[0])]

        for e in range(epoch):
            random.shuffle(sample_index)
            # loop through each training case and perform update
            for index in sample_index:
                u = int(train_data[index,0])
                i = int(train_data[index,1])
                
                u_time = train_data[index,3]
                i_time = u_time
                i_bin = (i_time-train_data[:,3].min())//bin_width
                
                r_ui = train_data[index,2]
                
                bias_u = self._user_bias[u-1]
                bias_i = self._item_bias[tmp_movie_dict[i]]
                
                bias_u_time = self._user_time_bias[(self._user_time_bias['userId']==u)&(self._user_time_bias['timestamp']==u_time)]['user_time_bias'].iloc[0]
                bias_i_time = self._item_bin_bias[i,i_bin]
                
                e_ui = r_ui - total_mean - bias_u - bias_u_time - bias_i - bias_i_time - np.dot(item_latent[:,self._movie_dict[i]].T, user_latent[:,u-1])

                grad_user_latent = e_ui * user_latent[:,u-1] - lam * item_latent[:,self._movie_dict[i]]
                if (all(np.abs(grad) > stopping_deriv for grad in grad_user_latent)):
                    item_latent[:,self._movie_dict[i]] = item_latent[:,self._movie_dict[i]] + lrate * grad_user_latent

                grad_item_latent = e_ui * item_latent[:,self._movie_dict[i]] - lam * user_latent[:,u-1]
                if (all(np.abs(grad) > stopping_deriv for grad in grad_item_latent)):
                    user_latent[:,u-1] = user_latent[:,u-1] + lrate * grad_item_latent

                grad_user_bias = e_ui - lam * bias_u
                if (np.abs(grad_user_bias) > stopping_deriv):
                    self._user_bias[u-1] = bias_u + lrate * grad_user_bias

                grad_item_bias = e_ui - lam * bias_i
                if (np.abs(grad_item_bias) > stopping_deriv):
                    self._item_bias[tmp_movie_dict[i]] = bias_i + lrate * grad_item_bias
                    
                grad_user_time_bias = e_ui - lam * bias_u_time
                if (np.abs(grad_user_time_bias) > stopping_deriv):
                    self._user_time_bias[(self._user_time_bias['userId']==u)&(self._user_time_bias['timestamp']==u_time)]['user_time_bias'].replace(bias_u_time, bias_u_time + lrate * grad_user_time_bias)
                    
                grad_item_time_bias = e_ui - lam*bias_i_time
                if (np.abs(grad_item_time_bias) > stopping_deriv):
                    self._item_bin_bias[i,i_bin] = bias_i_time + lrate * grad_item_time_bias

            self._user_latent = user_latent
            self._item_latent = item_latent
            
            self._model = 'dynamic'
    
    
    
    
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
        self._sim = pairwise_distances(self._item_latent.T,metric='cosine')
        self._sim = pd.DataFrame(self._sim)
        #self._sim.index = self._item_latent.columns
        #self._sim.columns = self._item_latent.columns
        if self._model == 'basic':
            r_ij = np.dot(self._item_latent.T, self._user_latent)
        elif self._model == 'bias':
            r_ij = self._total_mean + self._user_bias + np.dot(self._item_latent.T, self._user_latent)
            r_ij = (r_ij.T + self._item_bias).T
        elif self._model == 'dynamic':
            user_time_mean = self._user_time_bias.groupby('userId').mean()['user_time_bias'].tolist()
            item_time_mean = self._item_bin_bias.groupby('movieId').mean()
            r_ij = self._total_mean + self._user_bias + np.array(user_time_mean) + np.dot(self._item_latent.T, self._user_latent)
            r_ij = (r_ij.T + self._item_bias +np.array(item_time_mean)).T 
        # compute distance from test point to all train point
        #all_distance = [pairwise_distances(train_point, test_point, metric='cosine') for train_point in self._user_latent]
    
        # get nearest k neighbors' index
        #k_neighbors_class = np.argsort(all_distance)[:self._k]
        knn_r_ij = []
        tmp = list(test_point['movieId'].unique())
        for i in tmp:
            k_neighbors_class = np.argsort(self._sim[self._movie_dict[i]])[1:1+k]
            knn_r_ij.append(np.mean(r_ij[k_neighbors_class,:],axis=0))
        knn_r_ij = pd.DataFrame(knn_r_ij)
        knn_r_ij.index = self._data['movieId'].unique().tolist()
        knn_r_ij.columns = self._data['userId'].unique().tolist() 
        # get nearest k neighbors most frequent label
        # DON'T KNOW WHAT IS THE RETURN
        #return #Counter(self._y_train[k_neighbors_class]).most_common()[0][0]
        self._est_rating = knn_r_ij
        return knn_r_ij
    
    def predict(self,train_data = None, test_data = None):

        if train_data is not None and test_data is not None:
            self._train_data = train_data
            self._test_data = test_data

        train_RMSE = []
        test_RMSE = []
        
        if self._est_rating is not None:
            est_rating = self._est_rating
        else:
            est_rating = np.dot(self._item_latent.T, self._user_latent)
            
        est_rating = pd.DataFrame(est_rating)
        est_rating.index = self._data['movieId'].unique().tolist()
        est_rating.columns = self._data['userId'].unique().tolist()

        train_RMSE_cur = RMSE(np.array(self._train_data), est_rating)
        train_RMSE.append(train_RMSE_cur)
        print("training RMSE:", train_RMSE_cur)
        test_RMSE_cur = RMSE(np.array(self._test_data), est_rating)
        test_RMSE.append(test_RMSE_cur)
        print("test RMSE:", test_RMSE_cur)

        return [train_RMSE, test_RMSE]

