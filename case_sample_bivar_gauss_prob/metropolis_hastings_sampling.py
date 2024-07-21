# -*- coding: utf-8 -*-
"""
Created on 2024/07/21 13:13:29

@File -> metropolis_hastings_sampling.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Metropolis-Hastings采样
"""

from typing import List
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt
from case_sample_bivar_gauss_prob.bivar_mixed_gaussian_pdf import cal_merged_gauss_pdf


class MetropolisHastingsSampler(object):
    """Metropolis-Hastings采样器"""
    
    def __init__(self, bounds: List[List[float]]) -> None:
        self.bounds = bounds
        self.dim = 2
        assert len(self.bounds) == self.dim
        self.samples_arr = np.array([])
    
    def propose_a_sample(self) -> np.ndarray:
        """产生一个提议样本"""
        sample = np.zeros(self.dim)
        for dim in range(self.dim):
            sample[dim] = np.random.uniform(self.bounds[dim][0], self.bounds[dim][1])
        return sample
    
    def initialize_a_sample(self, sample: np.ndarray = None) -> np.ndarray:
        """初始化一个样本"""
        if sample is None:
            sample = self.propose_a_sample()
        return sample
    


if __name__ == "__main__":
    
    # ---- 参数设置 ---------------------------------------------------------------------------------
    
    # 待采样分布参数
    mu_lst = [np.array([-3, -3]), np.array([3, 3])]
    sigma_lst = [np.array([[1, 0.9], [0.9, 1]]), np.array([[1, -0.9], [-0.9, 1]])]
    weights = [0.1, 0.9]
    
    # 采样边界参数
    x_bounds = [-30, 30]
    y_bounds = [-30, 30]
    
    # ---- 采样 ------------------------------------------------------------------------------------
    
    bounds = [x_bounds, y_bounds]
    
    self = MetropolisHastingsSampler(bounds)
    
    n_loops = 100000
    p_last: float = None
    sample_init = np.array([-30, -30])
    for loop in range(n_loops):
        if loop == 0:
            sample_i = self.initialize_a_sample(sample_init)
            
            # 提议样本加入样本集
            self.samples_arr = sample_i.reshape(1, -1)
            
            # 计算提议样本的目标分布概率
            p = cal_merged_gauss_pdf(sample_i[0], sample_i[1], mu_lst, sigma_lst, weights)
            p_last = p
        else:
            sample_i = self.propose_a_sample()
            
            # 在上一步样本的基础上，按照二维高斯分布生成新样本
            sample_j = self.samples_arr[-1, :]
            
            # 根据转移概率新提议一个样本
            # 拒绝率很高的提议
            # sample_i = self.propose_a_sample()
            
            # 新提议样本以上一步样本为中心，按照二维高斯分布生成
            std_pos = 10
            sample_i = np.random.multivariate_normal(sample_j, np.eye(self.dim) * std_pos)
            
            # 相邻两步之间的转移概率
            Qij = 1
            Qji = 1
            
            # 计算提议样本的目标分布概率
            p = cal_merged_gauss_pdf(sample_i[0], sample_i[1], mu_lst, sigma_lst, weights)
            
            # 计算接受概率
            alpha = min([1, p / p_last * Qji / Qij])
            
            # 接受拒绝
            if np.random.rand() < alpha:
                self.samples_arr = np.vstack([self.samples_arr, sample_i])
                p_last = p
                
    # ---- 绘制采样结果 -----------------------------------------------------------------------------
    
    # 待采样分布
    # x = np.linspace(x_bounds[0], x_bounds[1], 100)
    # y = np.linspace(y_bounds[0], y_bounds[1], 100)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros(X.shape)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i, j] = cal_merged_gauss_pdf(X[i, j], Y[i, j], mu_lst, sigma_lst, weights)
    # plt.contour(X, Y, Z, 50, zorder=-1, linewidths=0.2, alpha=0.5)
    
    plt.scatter(self.samples_arr[:, 0], self.samples_arr[:, 1], marker="o", c = "k", s = 1, linewidths=0.2, alpha=1, zorder=1)
    
    # 将相邻两个样本之间连线
    for i in range(1, self.samples_arr.shape[0]):
        plt.plot([self.samples_arr[i - 1, 0], self.samples_arr[i, 0]], [self.samples_arr[i - 1, 1], self.samples_arr[i, 1]], c = "k", alpha=0.3, linewidth = 0.2, zorder=-2)
    
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()
    
    # ---- 绘制马尔可夫链 ----------------------------------------------------------------------------
    
    plt.figure(figsize = (5, 5))
    plt.plot(self.samples_arr[:, 0], c = "blue", linewidth = 0.5)
    plt.plot(self.samples_arr[:, 1], c = "orange", linewidth = 0.5)