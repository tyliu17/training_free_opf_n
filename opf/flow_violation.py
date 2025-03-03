import numpy as np

def compute_line_flow_violation(S_isf, p_inj, p_inj_true, f_max_numpy):
    n_line, n_sample = S_isf.shape[0], p_inj.shape[1]

    # 計算潮流
    flow_est = np.dot(S_isf, p_inj)
    flow_est0 = np.dot(S_isf, p_inj_true)

    # 計算違規
    f_binary = (np.abs(flow_est) - f_max_numpy > 0)
    f_binary0 = (np.abs(flow_est0) - f_max_numpy > 0)

    # 計算違規比例
    f_tot_sample = n_line * n_sample
    violation_rate_pred = np.sum(f_binary) / f_tot_sample
    violation_rate_true = np.sum(f_binary0) / f_tot_sample

    return violation_rate_pred, violation_rate_true
