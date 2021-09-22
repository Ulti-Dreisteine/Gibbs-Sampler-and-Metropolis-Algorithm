# Gibbs-Sampler-and-Metropolis-Algorithm
*Dreisteine, 2021.09.22*

## **一、算法背景**

1. 在工业过程建模、分子动力学模拟等领域，过程高维、复杂，难以获得足够样本反映其全貌。尝试使用迭代采样方式逐渐获得近似分布
2. 现有的抽样算法都是基于均匀分布等简单分布，我们尝试使用已有的简单的抽样方法对复杂过程进行采样

先介绍几个概念：

1. 系统状态$s$：  
    系统中所有$n$个变量的取值组合，即

    $$
    s_i = [x_{1,i}, x_{2,i}, \cdots, x_{n,i}] \tag{1}
    $$

2. 状态概率$\pi$：  
    系统处于状态$s_i$的概率为$\pi_i$

3. 状态转移概率矩阵$\bf P$：  
    如果将过程在不同状态之间的转移视作一个随机过程，其状态转移使用概率矩阵
    $\bf P$表示，其中$P_{i,j}$表示过程从状态$i$转移至状态$j$的概率，则有

    $$
    \sum_{j=1}^{m} P_{i,j} = 1, \forall i \in [1, m] \tag{2}
    $$

4. 系统状态转移Markov过程：  
    若系统在$t-1$采样步的状态为$s^{(t-1)}$，下一步的状态$s^{(t)}$仅取决于$s^{(t-1)}$和状态转移矩阵$\bf P$，而与更早的系统状态无关，即$Pr(s^{(t)}|s^{(t-1)},s^{(t-2)},\cdots) = Pr(s^{(t)}|s^{(t-1)})$，易有Markov链：

    $$
    s^{(0)} \rightarrow s^{(1)} \rightarrow \cdots \rightarrow s^{(t-1)} \rightarrow s^{(t)} \rightarrow \cdots \tag{3}
    $$

5. Markov链的平稳性：  
    对于一个非周期的Markov链，其任意两个状态保持连通，即经过有限次转移可达，则以下极限存在：

    $$
    \lim_{t\rightarrow\infty} P_{i,j}^t = \pi_j \tag{4}
    $$

    $$
    \lim_{t\rightarrow\infty} {\bf P}^{t} = \left[
        \begin{array}{}
        &\pi_1 &\pi_2 &\cdots &\pi_m \\ \tag{5}
        &\pi_1 &\pi_2 &\cdots &\pi_m \\
        &\cdots &\cdots &\cdots &\cdots \\
        &\pi_1 &\pi_2 &\cdots &\pi_m \\
        \end{array}
        \right]
    $$

    $$
    \pi_j = \sum_{i=1}^m \pi_i P_{i, j} \tag{6}
    $$

6. 方程$\pi{\bf P} = \pi$存在唯一非负解$\pi^*$，$\pi^*$即为该Markov链的平稳分布[^1]

7. Markov链的细致平稳条件：
    细致平稳条件是一种更为严格的平稳条件

    $$
    \pi_i \cdot P_{i,j} = \pi_j \cdot P_{j,i} \tag{7}
    $$
8. **注意**：通过简单尝试可知，对于特定$\pi$，满足细致平稳条件的$\bf P$并不唯一

个人体会和总结：

* 如果系统对应于一个平稳的Markov过程，那么将存在唯一的平稳分布概率$\pi$对应$\bf P$

* 对于具有很高维数$m$的复杂过程，$\pi$可以从样本获得，但是${\bf P}_{m \times m}$却较难估计

* 在第$t$步迭代中，已知前一步状态为$s^{(t-1)}=s_i$，下一步迭代时只需通过机理推导(分子动力学模拟)或数学构造等方式获得$\bf P$中第$i$行信息即可进行该步迭代

* 如果通过某种非机理方式构造出转移概率矩阵$\bf Q$，在过程样本上满足细致平稳条件，则可使用构造所得Markov链对过程进行采样。注意根据第8点结论，$\bf Q$可能并不等于系统本身的$\bf P$，因此Metropolis采集样本只满足细致平稳，不能用于过程预测等建模。

## **二、Metropolis采样算法**

对于一个十分复杂的过程，其中变量分布为$\pi$并**潜在地**对应着一个转移概率矩阵$\bf P$。我们希望利用一个有着**对称**转移矩阵${\bf Q}$的简单分布$\theta$进行采样，以一定概率接受采得样本作为对$(\pi, \bf P)$过程的近似细致平稳分布采样，用于其他统计分析。

首先，给出一个对称的提议转移概率矩阵$\bf Q$，但该矩阵一般不满足式(7)中的细致平稳条件，即

$$
\pi_i \cdot Q_{i,j} \neq \pi_j \cdot Q_{j,i} \tag{8}
$$

为了构造细致平稳条件，对上式左右分别乘以接受概率$\alpha$：

$$
\pi_i \cdot Q_{i,j} \cdot \alpha_{i,j} = \pi_j \cdot Q_{j,i} \cdot \alpha_{j,i} \tag{9}
$$

使得接受概率$\alpha$与提议矩阵$\bf Q$的组合满足如下概率转移矩阵：

$$
\begin{align*}
{\hat P}_{i,j} = Q_{i,j} \cdot \alpha_{i, j} \\ \tag{10}
{\hat P}_{j,i} = Q_{j,i} \cdot \alpha_{j, i} \\
\end{align*}
$$

这样一来，所构造的转移概率$\hat P$满足细致平稳条件：

$$
\pi_i \cdot {\hat P}_{i,j} = \pi_j \cdot {\hat P}_{j,i} \tag{11}
$$

在Metropolis算法中，假设我们在第$t-1$步获得状态样本$x^{(t-1)}=s_i$，接下来便从$\theta(s|s_i)$中新采集一个样本$\hat s$。以概率$\alpha = \min \{1, \pi(\hat s) / \pi(s_i)\}$接受$x^{(t)} = \hat s$，否则$x^{(t)} = x^{(t-1)}=s_i$。该过程对应的转移概率满足：

$$
\hat P_{i,j} = \alpha_{i,j} \cdot Q_{i,j} \tag{12}
$$

那么，

$$
\begin{align*}
\pi_i \cdot \hat P_{i,j} &= \pi_i \cdot \alpha_{i,j} \cdot Q_{i,j} \\ \tag{13}
&= \pi_i \cdot \min \{1, \pi_j / \pi_i\} \cdot Q_{i,j} \\
&= \min\{\pi_i \cdot Q_{i,j}, \pi_j \cdot Q_{i,j}\} \\
&= \min\{\pi_i \cdot Q_{j,i}, \pi_j \cdot Q_{j,i}\} \\ 
&= \pi_j \cdot \min\{1, \pi_i / \pi_j\} \cdot Q_{j,i} \\
&= \pi_j \cdot \hat P_{j,i} \\
\end{align*}
$$

因此，如上方式构造所得状态转移矩阵$\bf \hat P$满足细致平稳条件。根据讨论6，按照该方式获得的采样分布即为$\pi$[^2]。

![1](img/Metropolis采样.png)  


## **三、Metropolis Hastings采样算法**

Metropolis算法中的接受概率$\alpha$可能会很小，导致算法需要经历很多次迭代才能到达平稳。因此，MH算法考虑把式(9)中等式两边的接受概率同步放大，将其中的一个接受概率设置为1，这样就能保证每次迭代过程中接收新状态的概率越大，加速算法收敛。

$$
\alpha_{j,i} = \min\{1, \frac{\pi_j Q_{i,j}}{\pi_i Q_{j,i}}\}
$$

## **四、总结**

1. 对于高维复杂系统，仅需知道$\pi_i, \pi_j$的绝对值计算方式或相对值$\pi_i /\pi_j$便可对系统进行采样
2. Metropolis采样算法与时序系统无关，采样过程中样本的变化也与时间变化无关，不要混淆


[^1]: [Markov过程平稳性讨论](https://www.cnblogs.com/coshaho/p/9740937.html)
[^2]: [Metropolis算法](https://blog.csdn.net/lin360580306/article/details/51240398)







