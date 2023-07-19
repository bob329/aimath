#16-2.py

#����pandas���ݹ��߼�
import pandas as pd
#�������ѧϰ���еĹ�һ������
from sklearn.preprocessing import normalize
#�����ξ��ຯ����������״ͼ����
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
#������������ʾ���ƿ�
from matplotlib.ticker import MultipleLocator

%matplotlib inline

scores_df = pd.read_csv('./datasets/eurovision-2016-televoting.csv', index_col=0)
country_names = list(scores_df.index)
scores_df.head()

#ȱʧֵ��䣬û�еľ��Ȱ��������
scores_df = scores_df.fillna(12)

#��һ��
samples = normalize(scores_df.values)

plt.figure(figsize=(10,12),dpi=80)
plt.subplots_adjust(hspace=0.5)

#single method distance clustering
mergings = linkage(samples, method='single')
p1=plt.subplot(211)
yminorLocator = MultipleLocator(0.05) 
p1.yaxis.set_minor_locator(yminorLocator)
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=10,
)
p1.set_title("single-min distance",fontsize=18)

#complete method distance clustering
mergings = linkage(samples, method='complete')
p2=plt.subplot(212)
yminorLocator = MultipleLocator(0.05) 
p2.yaxis.set_minor_locator(yminorLocator)
dendrogram(mergings,
           labels=country_names,
           leaf_rotation=90,
           leaf_font_size=10,
)
p2.set_title("complete-max distance",fontsize=18)

plt.show()