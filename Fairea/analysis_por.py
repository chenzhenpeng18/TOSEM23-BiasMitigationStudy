import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Fairea.utility import get_data,write_to_file
from Fairea.fairea import create_baseline,normalize,get_classifier,classify_region,compute_area
from shapely.geometry import Polygon, Point, LineString
from matplotlib import pyplot as plt
import matplotlib
import argparse

base_points = {}
for i in ['rf','lr','svm']:
    base_points[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        base_points[i][j]={}

base_points_key1 = ['SPD','AOD','EOD', 'ERD', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    (dataset_pre,dataset_aft) = dataset.lower().split('-')
    if dataset == 'Mep-Race':
        dataset_pre = 'mep'
        dataset_aft = 'RACE'
    fin = open('baseline/'+dataset_pre+'_'+dataset_aft+'_baseline','r')
    count = 0
    for line in fin:
        if count %11 == 0 or count %11 ==4:
            count += 1
            continue
        if int(count/11) == 0:
            key1 = 'lr'
        elif int(count/11) == 1:
            key1 = 'rf'
        elif int(count/11) == 2:
            key1 = 'svm'
        base_points[key1][dataset][base_points_key1[count%11-1]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        count+=1
    fin.close()

data = {}
for i in ['rf','lr','svm']:
    data[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        data[i][j]={}
        for k in ['F-P','F-R','F-F1','UnF-P','UnF-R','UnF-F1','Acc','Mac-P','Mac-R','Mac-F1','AUC', 'MCC','SPD','AOD','EOD','ERD']:
            data[i][j][k]={}

data_key_value = {1:'Acc', 2:'F-R', 3:'UnF-R', 4:'Mac-R', 5:'F-P', 6:'UnF-P', 7:'Mac-P', 8:'F-F1', 9:'UnF-F1', 10:'Mac-F1', 11: 'AUC', 12:'MCC', 13:'SPD', 15:'AOD', 16:'EOD',17:'ERD'}

for j in ['rf','lr','svm']:
	for name in ['fairsmote','fairway','ad','eqo','pr','mf1','mf2','cpp1','cpp2','cpp3','default','di','lfr','rew','roc1','roc2','roc3']:
		for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
			(dataset_pre,dataset_aft) = dataset.lower().split('-')
			if dataset == 'Mep-Race':
				dataset_pre = 'mep'
				dataset_aft = 'RACE'
			fin = open('../Result/'+name+'_'+j+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
			count = 0
			for line in fin:
				count=count+1
				if count in data_key_value:
					data[j][dataset][data_key_value[count]][name]=line.strip().split('\t')[1:51]
			fin.close()

	for name in ['op']:
		for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age']:
			(dataset_pre,dataset_aft) = dataset.lower().split('-')
			if dataset == 'Mep-Race':
				dataset_pre = 'mep'
				dataset_aft = 'RACE'
			fin = open('../Result/'+name+'_'+j+'_'+dataset_pre+'_'+dataset_aft+'.txt','r')
			count = 0
			for line in fin:
				count=count+1
				if count in data_key_value:
					data[j][dataset][data_key_value[count]][name]=line.strip().split('\t')[1:51]
			fin.close()
		for mm in ['F-P','F-R','F-F1','UnF-P','UnF-R','UnF-F1','Acc','Mac-P','Mac-R','Mac-F1','AUC', 'MCC','SPD','AOD','EOD','ERD']:
			data[j]['Bank-Age'][mm][name]='/'
			data[j]['Mep-Race'][mm][name]='/'

region_count = {}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['AUC', 'Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in ['rf','lr','svm']:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in ['op','lfr','rew','di','pr','ad','mf1','mf2','roc1','roc2','roc3','cpp1','cpp2','cpp3','eqo','fairway','fairsmote']:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in ['rf','lr','svm']:
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        for fairmetric in ['SPD','AOD','EOD']:
            for permetric in ['AUC', 'Acc','Mac-P','Mac-R','Mac-F1','MCC']:
                for name in ['op', 'lfr', 'rew', 'di', 'pr', 'ad', 'mf1', 'mf2', 'roc1', 'roc2', 'roc3', 'cpp1', 'cpp2',
                             'cpp3', 'eqo', 'fairway', 'fairsmote']:
                    methods = dict()
                    name_fair50 = data[i][j][fairmetric][name]
                    name_per50 = data[i][j][permetric][name]
                    if name_fair50 == '/':
                        continue
                    for count in range(50):
                        methods[str(count)] = (float(name_per50[count]), float(name_fair50[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1


fout = open('region_result','w')

name_real = {'op': 'OP', 'lfr':'LFR', 'rew':'RW', 'di':'DIR', 'pr':'PR', 'ad':'AD', 'mf1':'MFC-FDR', 'mf2':'MFC-SR', 'roc1':'ROC-SPD', 'roc2':'ROC-AOD', 'roc3':'ROC-EOD', 'cpp1':'CEO-FNR',
                             'cpp2':'CEO-FPR', 'cpp3':'CEO-W', 'eqo':'EOP', 'fairway':'Fairway', 'fairsmote':'Fair-SMOTE'}
for name in ['op', 'lfr', 'rew', 'di', 'pr', 'ad', 'mf1', 'mf2', 'roc1', 'roc2', 'roc3', 'cpp1',
                             'cpp2', 'cpp3', 'eqo', 'fairway', 'fairsmote']:
    final_count = {}
    fout.write(name_real[name])
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for permetric in ['AUC', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age',
                          'Bank-Age', 'Mep-Race']:
                for i in ['rf', 'lr', 'svm']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    final_sum = 0
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')

fout.write('\n')
fout.write('\n')
fout.write('\n')

for i in ['lr', 'svm', 'rf']:
    final_count = {}
    fout.write(i)
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for permetric in ['AUC', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for name in ['op', 'lfr', 'rew', 'di', 'pr', 'ad', 'mf1', 'mf2', 'roc1', 'roc2', 'roc3', 'cpp1', 'cpp2',
                     'cpp3', 'eqo', 'fairway', 'fairsmote']:
                for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age',
                          'Bank-Age', 'Mep-Race']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    final_sum = 0
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')


fout.write('\n')
fout.write('\n')
fout.write('\n')


for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age',
                          'Bank-Age', 'Mep-Race']:
    final_count = {}
    fout.write(j)
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for i in ['rf', 'lr', 'svm']:
        for permetric in ['AUC', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
            for fairmetric in ['SPD', 'AOD', 'EOD']:
                for name in ['op', 'lfr', 'rew', 'di', 'pr', 'ad', 'mf1', 'mf2', 'roc1', 'roc2', 'roc3', 'cpp1', 'cpp2',
                             'cpp3', 'eqo', 'fairway', 'fairsmote']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    final_sum = 0
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')

fout.write('\n')
fout.write('\n')
fout.write('\n')

for fairmetric in ['SPD', 'AOD', 'EOD']:
    for permetric in ['AUC', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC']:
        final_count = {}
        fout.write(fairmetric + '&' + permetric)
        for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
            final_count[region_kind] = 0
        for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age',
                      'Bank-Age', 'Mep-Race']:
            for i in ['rf', 'lr', 'svm']:
                for name in ['op', 'lfr', 'rew', 'di', 'pr', 'ad', 'mf1', 'mf2', 'roc1', 'roc2', 'roc3', 'cpp1',
                             'cpp2', 'cpp3', 'eqo', 'fairway', 'fairsmote']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
        final_sum = 0
        for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
            final_sum += final_count[region_kind]
        for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
            fout.write('\t%f' % (final_count[region_kind]/final_sum))
        fout.write('\n')

fout.close()


