#16-1.py

#�����ξ��ຯ����������״ͼ����
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
#������������ʾ���ƿ�
from matplotlib.ticker import MultipleLocator
#�������ݲ������߼�
import pandas as pd

%matplotlib inline
#��ȡ���ݼ�
seeds_df = pd.read_csv('./datasets/seeds-less-rows.csv')
seeds_df.head()

#ȥ����ʶ�м������
varieties = list(seeds_df.pop('grain_variety'))
samples = seeds_df.values

#���в�ξ���
mergings = linkage(samples, method='complete')

#��״ͼ���
plt.figure(figsize=(10,6),dpi=80)
ax=plt.subplot(111)
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=10,
)
yminorLocator = MultipleLocator(0.2) 
ax.yaxis.set_minor_locator(yminorLocator)
plt.show()
