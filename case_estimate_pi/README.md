### 基于随机采样方法对圆周率 $\pi$ 进行估计。

#### 一、蒙特卡洛近似

下图中 $x_i$ 表示在第 $i$ 次随机试验中产生的灰色正方形 $1\times 1$ 内的一个随机点，$f(x_i)$ 表示 $x_i$ 是否位于蓝色曲线表示的四分之一圆内：若位于圆内，则 $f(x_i)=1$；否则 $f(x_i)=0$。

<img src="img/fig_1.PNG" width="200">


首先，由蓝色曲线面积 $S_{\rm circle}$ 和灰色方框面积 $S_{\rm square}$ 对比可知：

$$
\begin{align*}
    \frac{S_{\rm circle}}{S_{\rm square}} &= \frac{0.25\pi}{1} \tag{1} \\
    &= \lim_{N\rightarrow\infty} \frac{N_{f(x)=1}}{N_{f(x)=1} + N_{f(x)=0}} \tag{2} \\
    &= \mathbb E_x[f(x)] \tag{3} \\
    &\approx \frac{1}{N}\sum_{i=1}^{N}f(x_i) \text{, where } x_1 \cdots x_N \stackrel{i.i.d.}\sim {\rm Uniform}(0, 1) \tag{4} 
\end{align*}
$$

接下来，编写代码：

```python
def estimate_pi(N: int) -> float:
    x = np.random.uniform(0, 1, (N, 2))
    count = np.sum(np.sum(x**2, axis=1) <= 1)
    pi_est = 4 * count / N
    return pi_est
```

分别计算在不同采样量```N```下：
1. 所得 $\pi$ 的估计结果均值和标准差变化；
2. 对应估计残差的绝对值随采样量变化。

如下图所示：

<img src="img/fig_monte_carlo_approximation.png" width="500">

#### 二、重要性采样

对于式(3)，

$$
\begin{align*}
    {\mathbb E_{x}}[f(x)] &= \int_{x}f(x)p(x){\rm d}x \tag{5} \\
\end{align*}
$$

引入另一个分布 $q(x)$，有：

$$
\begin{align*}
    {\mathbb E_{x}}[f(x)] &= \int_{x}f(x)\frac{p(x)}{q(x)}q(x){\rm d}x \tag{6} \\
    &\approx \frac{1}{N}\sum_{i=1}^{N}f(x_i)\frac{p(x_i)}{q(x_i)} \text{, where } x_1 \cdots x_N \stackrel{i.i.d.}\sim q(x) \tag{7}
\end{align*}
$$

其中，$p(x_i)/q(x_i)$ 又名重要性值，即不同 $f(x_i)$ 对于整体期望 ${\mathbb E_{x}}[f(x)]$ 的影响权重。

接下来，考虑采用重要性采样对圆周率 $\pi$ 进行估计。这里选择 $q(x) \sim {\rm Normal}(\mu_q=0.5, \sigma_q=0.1)$

<img src="img/fig_importance_sampling.png" width="500">