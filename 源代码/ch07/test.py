import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
def Discrete_pmf():
    xk = np.arange(5) # 所有可能的取值[0 1 2 3 4]
    pk = (1/16, 1/4, 3/8, 1/4, 1/16)  # 各个取值的概率
    #用rv_discrete 类自定义离散概率分布rvs
    dist = stats.rv_discrete(name='custm', values=(xk, pk))
    #调用其rvs方法，获得符合概率的100个随机数:
    rv=dist.rvs(size=100)
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
    #显示概率函数
    ax0.set_title("概率函数")
    ax0.plot(xk, pk, 'ro', ms=8, mec='r')
    ax0.vlines(xk, 0, pk, colors='r', linestyles='-', lw=2)
    for i in xk:
        ax0.text(i,pk[i],'%.3f'%pk[i],ha='center',va='bottom')
    #显示"分布函数"
    ax1.set_title("分布函数")
    pk1=dist.cdf(xk)
    #利用直方图显示分布函数
    ax1.hist(rv,4,density=1, histtype='step', facecolor='blue'\
             ,alpha=0.75,cumulative=True,rwidth=0.9)
    for i in xk:
        ax1.text(i, pk1[i],'%.3f'%pk1[i],ha='center',va='bottom')
    plt.show()    
if __name__=='__main__':
    Discrete_pmf()