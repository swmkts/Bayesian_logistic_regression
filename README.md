# Bayesian_logistic_regression
## This is the implementation of Bayesian logistic regression.

<span style="color:red">The following has not completed yet.<\span>

To infer the posterior, I use some methods.
1. Metropolis Hastings
2. Hamiltonian Monte Carlo
3. No U Turn Sampler(NUTS)
4. Variational Bayesian Inference

# Non-Bayesian logistic regression optimization methods(sklearn based arguments)
1. ニュートン法(`newton_cg`)
2. L-BFGSアルゴリズム(準ニュートン法)(`lbfgs`)
3. A library for Large Linear Classification(座標降下法に基づく)(`liblinear`)
4. Stochastic Average Gradient(SGDの改善)
5. A Fast Incremental Gradient method with Support for Non-Strongly Convex Composite Objectives

+ Stochastic Average Descent  
+ 
These method should be selected according to the methods of regularization.
Show (https://scikit-learn.org/stable/modules/linear_model.html）

You can use both bayesian and non bayesian method of logistic regression. You can choose which you use via `--method` option.

# Installation


# Usage
Options:
 -h [--help]    show help
 -m [--method]  which method you want to use
 



# 変分推論の導出
- ポイント
1. 全微分、偏微分などの関係をしっかりと把握して目的関数の最適化を行う。
2. 

## 事後分布
出力の2値ベクトル$\boldsymbol{Y}$は以下のようなベルヌーイ分布により出力される。
$$
p(\boldsymbol{Y}|\boldsymbol{X}, \boldsymbol{W}) = \prod_{n=1}^{N}Bern(\boldsymbol{y_n}|f(\boldsymbol{W}, \boldsymbol{x_n}))
$$
また、係数$\boldsymbol{W}$は平均0の等分散で独立なガウス分布から生じると仮定し、事前分布を設定する。
$$
p(\boldsymbol{W}) = \prod_{m=1}^{M} \prod_{d=1}^{D} N(w_{m,d}|0, \lambda_{-1})
$$
ベルヌーイ分布のパラメータとするため、$f$にはシグモイド関数を用いる。  
求めるべきものはパラメータ$W$の事後分布である。ベイズの定理により
$$
p(\boldsymbol{W}|\boldsymbol{X}, \boldsymbol{Y}) 
    = \frac{p(\boldsymbol{Y}|\boldsymbol{X}, \boldsymbol{W})p(\boldsymbol{W})}
    {p(\boldsymbol{X}, \boldsymbol{Y})}
$$
となる。

## 最適化問題の定式

いま、シグモイド関数により事後分布が解析的な確率分布として求められないので、近似を行う。
$$
q(\boldsymbol{W}; \eta) = \prod_{m=1}^{M} \prod_{d=1}^{D} N(w_{m,d}|\mu_{m,d}, \sigma_{m,d}^{2})
$$
と近似する。これと真の事後分布とが近くなるようなパラメータ$W$を求めるのが今回の目的である。  
最適化問題は以下のように定式化できる。
$$
\eta_{opt} = 
    \mathop{\rm argmin}\limits_{n} 
        KL[q(\boldsymbol{W}; \eta)||p(\boldsymbol{W}|\boldsymbol{X}, \boldsymbol{Y}]
$$
他のモデルで行うような平均場近似はシグモイド関数により行えない。よって、この最適化問題を解くために、KLdivergenceを$\eta$で微分することを考える。qとpのKLdivergenceを期待値表記すると
$$
KL[q(\boldsymbol{W}; \eta)||p(\boldsymbol{W}|\boldsymbol{X},\boldsymbol{Y}]
    = 
$$

## リパラメタライゼーショントリック
初めの2項はガウス分布に従う確率変数の平均なので解析的に求めることができる。しかし、第3項は、先ほど解析計算ができないために避けた尤度関数である。これはモンテカルロ法を用いれば計算は可能であるが、計算結果はもはや$\eta$の関数ではなくなってしまう。  
これは$eta$をパラメータとする確率分布からサンプリングをすることに起因するため、このパラメータをうまくサンプリングと引き離せないかと考えてやる。これが李パラメタライゼーショントリックと呼ばれる手法である。  
いま、$q$はガウス分布で近似していたので確率変数$\boldsymbol{W}$は
$$
w_{m,d} \sim N(\mu_{m,d}, \sigma_{m,d})
$$
により発生している。この状態では結果にパラメータの値は含まれない。  
$$
w_{m,d} = \mu_{m,d} + \sigma_{m,d}\tilde{\epsilon}
\\
\tilde{\epsilon} \sim N(0, 1)
$$
とすることにより、パラメータの既知の確率分布からの乱数を用いて$w$を表現することができ、パラメータ$\mu, \sigma$の関数として表すことができる。  
この手法は変分オートエンコーダ(VAE)でも使われている。  
よって、サンプルの一つの$\boldsymbol{W}$を用いてKLdivergenceを近似できる。
$$

$$
今回はガウス分布で近似したので、このKLdivergenceの近似を$\mu, \sigma$で微分することで勾配法を適用する。ポイントで述べたように、ここでは連鎖律に注意して慎重に計算を進める必要がある。  
必要な公式として、2変数関数$z = f(x, y)$を考える。いま、$x=g(u,v), y=h(u,v)$と表せるとする。この時、関数$f$の偏微分について以下が成り立つ。(http://www.f.waseda.jp/kinoue/mathandstat/[henbibun]gouseikansunobibun[2hensu].pdf)  
$$
\begin{aligned}
&\frac{\partial z}{\partial x}
    = \frac{\partial z}{\partial u} \frac{\partial u}{\partial x}
        + \frac{\partial z}{\partial v} \frac{\partial v}{\partial x}
\\
& \frac{\partial z}{\partial y}
    = \frac{\partial z}{\partial u} \frac{\partial u}{\partial y}
        + \frac{\partial z}{\partial v} \frac{\partial v}{\partial y}
\end{aligned}
$$
この関係を今考えている問題に当てはめると、
$$
x = w = g(\mu, \sigma) = \mu + \sigma\tilde{\epsilon}
\\
y = \mu = h(\mu, \sigma) = \mu + 0 \cdot \sigma
$$
とすればよく、この時、
$$
\frac{\partial w}{\partial \mu} = 1, \frac{}{}
$$