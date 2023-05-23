import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

trueProp = 0.01


case = pd.read_csv('results/rarect/%s/res1case.csv' % trueProp, header=None)
ctrl = pd.read_csv('results/rarect/%s/res1ctrl.csv' % trueProp, header=None)

r, c = case.shape


fraction = []
label = []
celltypes = []

for i in range(r):
    for j in range(c):
        fraction.append(case.iloc[i,j])
        fraction.append(ctrl.iloc[i,j])

        celltypes.append(i + 3)
        celltypes.append(i + 3)

        label.append('case')
        label.append('control')

df = pd.DataFrame({'est.prop': fraction, 'label': label, '#celltypes': celltypes})

sns.boxplot(x='#celltypes', y='est.prop', hue='label', data=df)
plt.axhline(trueProp, color='c', alpha=0.3)

plt.show()
