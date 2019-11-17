# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 00:13:14 2019

@author: Thinkpad
"""
import numpy as np
import pandas as pd
import multiprocessing
import time 
import os
os.chdir('D:\\CUSTAT\\5243\\fall2019-project4-sec1-grp1\\lib')
from Matrix_Factorization_A1 import Matrix_Factorization as mf
from Evaluation import RMSE
from sklearn.metrics.pairwise import pairwise_distances


data = pd.read_csv('../data/ml-latest-small/ratings.csv')
few_comments_idx = data['movieId'].value_counts()<5
few_comments_idx = few_comments_idx.index

#test_data = data.sample(frac =.99)
#train_data = data.drop(test_data.index)

test_data = data[(data['movieId'].isin(few_comments_idx))].sample(frac = .2)
train_data = data.drop(test_data.index)
test_data.shape[0],train_data.shape[0]

model = mf(data, train_data,test_data)

model.gradesc(f = 10,lam = 0.1,lrate = 0.1,maxiter = 10,stopping_deriv = 0.01)
model.KNN(k=5,test_point=data)

def KNN(k=5,test_point=None):
    sim = pairwise_distances(model._item_latent.T,metric='cosine')
    sim = pd.DataFrame(sim)
    sim.index = model._item_latent.columns
    sim.columns = model._item_latent.columns
        
    r_ij = np.dot(model._item_latent.T, model._user_latent)
    # compute distance from test point to all train point
    #all_distance = [pairwise_distances(train_point, test_point, metric='cosine') for train_point in self._user_latent]

    # get nearest k neighbors' index
    #k_neighbors_class = np.argsort(all_distance)[:self._k]
    knn_r_ij = []
    tmp = test_point['movieId'].unique()
    for i in range(len(tmp)):
        k_neighbors_class = np.argsort(sim[tmp[i]])[1:1+k]
        knn_r_ij.append(np.mean(r_ij[k_neighbors_class,:],axis=0))
    knn_r_ij = pd.DataFrame(knn_r_ij)
    knn_r_ij.index = model._item_latent.columns
    knn_r_ij.columns = model._user_latent.columns    
    # get nearest k neighbors most frequent label
    # DON'T KNOW WHAT IS THE RETURN
    #return #Counter(self._y_train[k_neighbors_class]).most_common()[0][0]
    return knn_r_ij

est_rating = KNN(k=5,test_point=data)

train_RMSE_cur = RMSE(model._train_data, est_rating)
print("training RMSE:", train_RMSE_cur)
test_RMSE_cur = RMSE(model._test_data, est_rating)
print("test RMSE:", test_RMSE_cur)

train_rmse1, test_rmse1 = model.predict(train_data,test_data)

start = time.time()
result2 = model.gradesc_bias(f = 10,lam = 0.1,lrate = 0.1,stopping_deriv = 0.04)
train_rmse2, test_rmse2 = model.predict(train_data,test_data)
finish = time.time()
run_time = finish - start
print('Run time is {:.2f}s'.format(run_time))

result3 = model.gradesc_dynamic(f = 10,lam = 0.1,lrate = 0.1,stopping_deriv = 0.04)
train_rmse3, test_rmse3 = model.predict(train_data,test_data)

