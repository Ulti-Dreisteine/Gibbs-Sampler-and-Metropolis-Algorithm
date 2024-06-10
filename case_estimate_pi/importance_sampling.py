from scipy.stats import norm
from sklearn.utils import resample
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_f(x):
    return 1 if np.sum(x**2) <= 1 else 0


def estimate_pi(N: int, x_total: np.ndarray, coeff_a: float) -> float:
    x = resample(x_total, n_samples=N, replace=True)
    px = np.ones(len(x))
    qx = norm.pdf(x[:, 0], mu_q, sigma_q) * 1.0 * coeff_a  # type: ignore
    f = np.apply_along_axis(cal_f, 1, x)
    pi_est = 4 * np.sum(f * px / qx) / N
    return pi_est  # type: ignore


if __name__ == "__main__":
    
    # ---- 构建样本 ----------------------------------------------------------------------------------
    
    N_total = int(1e8)
    mu_q, sigma_q = 0.5, 0.1
    x_a = np.random.normal(mu_q, sigma_q, N_total)
    x_b = np.random.uniform(0, 1, N_total)
    x_total = np.c_[x_a, x_b]
    
    # 只选出第一列在0到1之间的样本，因此需要对第一维的概率密度函数进行修正
    x_total = x_total[(x_total[:, 0] >= 0) & (x_total[:, 0] <= 1)]
    coeff_a = N_total / len(x_total)
    
    # ---- 估计圆周率 --------------------------------------------------------------------------------
    
    # TODO：将均值和标准差换为中位数和四分位数
    
    N_lst = [10, 100, 1000, 10000, 100000, 1000000]
    pi_est_mean_lst = []
    pi_est_std_lst = []
    error_mean_lst = []
    error_std_lst = []
    
    repeats = 10
    EPS = 1e-6
    
    for N in N_lst:
        
        pi_repeat_lst = []
        for i in range(repeats):
            pi_est = estimate_pi(N, x_total, coeff_a)
            pi_repeat_lst.append(pi_est)
        
        pi_est_mean_lst.append(np.mean(pi_repeat_lst))
        pi_est_std_lst.append(np.std(pi_repeat_lst))
        
        abs_errors = [np.abs(p - np.pi) + EPS for p in pi_repeat_lst]
        error_mean_lst.append(np.mean(abs_errors))
        error_std_lst.append(np.std(abs_errors))
    
    plt.figure(figsize=(6, 3))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(N_lst, pi_est_mean_lst, yerr=2*np.array(pi_est_std_lst), fmt="o-", label=r"estimated $\pi$")
    plt.axhline(np.pi, color="k", linestyle="--", label=r"true $\pi$")
    plt.legend(loc="best")
    plt.xscale("log")
    plt.xlabel(r"sample size $N$")
    plt.ylabel(r"estimated $\pi$")
    
    plt.subplot(1, 2, 2)
    plt.errorbar(N_lst, error_mean_lst, yerr=2*np.array(error_std_lst), fmt="o-")
    plt.axhline(0.0, color="k", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"sample size $N$")
    plt.ylabel(r"estimation error")
    
    plt.tight_layout()
    plt.savefig("img/fig_importance_sampling.png", dpi=450)
    