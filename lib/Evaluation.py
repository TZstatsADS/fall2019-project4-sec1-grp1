import numpy as np

def RMSE(rating, est_rating):
    sqr_error = []
    for r in range(rating.shape[0]):
        u = rating.iloc[r]['userId']
        i = rating.iloc[r]['movieId']
        r_ui = rating.iloc[r]['rating']
        est_r_ui = est_rating.loc[i, u]
        sqr_error.append((r_ui - est_r_ui) ** 2)
    return np.sqrt(np.mean(sqr_error))