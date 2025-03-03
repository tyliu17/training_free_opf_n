import numpy as np

def compute_error_metrics(y_test2, y_pred1, n_test):
    err_L2 = np.zeros(n_test)
    err_Linf = np.zeros(n_test)
    err_L2_v = np.zeros(n_test)
    err_Linf_v = np.zeros(n_test)

    for i in range(n_test):
        err_L2[i] = np.linalg.norm(y_test2[:, 0, i] - y_pred1[:, 0, i]) / np.linalg.norm(y_test2[:, 0, i])
        err_Linf[i] = np.max(np.abs(y_test2[:, 0, i] - y_pred1[:, 0, i])) / np.max(np.abs(y_test2[:, 0, i]))

        err_L2_v[i] = np.linalg.norm(y_test2[:, 1, i] - y_pred1[:, 1, i]) / np.linalg.norm(y_test2[:, 1, i])
        err_Linf_v[i] = np.max(np.abs(y_test2[:, 1, i] - y_pred1[:, 1, i])) / np.max(np.abs(y_test2[:, 1, i]))

    return np.mean(err_L2), np.mean(err_Linf), np.mean(err_L2_v), np.mean(err_Linf_v), np.std(err_L2), np.std(err_L2_v)   
