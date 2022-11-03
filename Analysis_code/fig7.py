import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

per_list = ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']
fair_list = ['SPD', 'AOD','EOD','ERD']

res_forplot = {}

fin = open('../Fairea_baseline/adult_lr_sex_baseline','r')
idx = 0
for line in fin:
    idx +=1
    if idx >= 1 and idx <=4:
        res_forplot[fair_list[idx-1]]=np.array(list(map(float,line.strip().split('\t')[1:])))
    else:
        res_forplot[per_list[idx-5]] = np.array(list(map(float, line.strip().split('\t')[1:])))

for i in  fair_list:
    for j in per_list:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 5),constrained_layout=True)
        axes.plot(res_forplot[i],res_forplot[j],color="#1874CD",marker = "^", linestyle = "solid",linewidth=3,markersize=10)
        axes.set_xlim(0)
        axes.set_xlabel(i,fontsize=20)
        axes.set_ylabel(j,fontsize=20)
        axes.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        axes.tick_params(labelsize=20)
        plt.savefig(i+j+".pdf")