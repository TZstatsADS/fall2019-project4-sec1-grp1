import numpy as np

def RMSE(rating, est_rating):
    sqr_error = []
    for r in range(rating.shape[0]):
        u = int(rating[r,0])
        i = int(rating[r,1])
        r_ui = rating[r,2]
        est_r_ui = est_rating.loc[i, u]
        sqr_error.append((r_ui - est_r_ui) ** 2)
    return np.sqrt(np.mean(sqr_error))