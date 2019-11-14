import pandas as pd
from Matrix_Factorization_A1 import Matrix_Factorization as mf

data = pd.read_csv('../data/ml-latest-small/ratings.csv')

test_data = data.sample(frac =.99)
train_data = data.drop(test_data.index)
model = mf(data, train_data,test_data)
# result1 = model.gradesc(f = 10,lam = 0.1,lrate = 0.1,stopping_deriv = 0.04)
# train_rmse1, test_rmse1 = model.predict(train_data,test_data)
# result2 = model.gradesc_bias(f = 10,lam = 0.1,lrate = 0.1,stopping_deriv = 0.04)
# train_rmse2, test_rmse2 = model.predict(train_data,test_data)

result3 = model.gradesc_dynamic(f = 10,lam = 0.1,lrate = 0.1,stopping_deriv = 0.04)
train_rmse3, test_rmse3 = model.predict(train_data,test_data)
