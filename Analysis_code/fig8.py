import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numpy import mean
from Fairea.fairea import normalize,classify_region,compute_area
from shapely.geometry import LineString

base_points = {}
for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    base_points[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        base_points[i][j]={}

base_points_key1 = ['SPD','AOD', 'EOD', 'ERD', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    (dataset_pre,dataset_aft) = dataset.lower().split('-')
    if dataset == 'Mep-Race':
        dataset_pre = 'mep'
        dataset_aft = 'RACE'
    for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
        fin = open('../Fairea_baseline/'+dataset_pre+'_'+i+'_'+dataset_aft+'_baseline','r')
        count = 0
        for line in fin:
            count+=1
            base_points[i][dataset][base_points_key1[count-1]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()

data = {}
for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    data[i]={}
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        data[i][j]={}
        for k in ['F-P','F-R','F-F1','UnF-P','UnF-R','UnF-F1','Acc','Mac-P','Mac-R','Mac-F1','AUC', 'MCC','SPD','AOD','EOD','ERD']:
            data[i][j][k]={}

data_key_value = {1: 'Acc', 2: 'F-R', 3: 'UnF-R', 4: 'Mac-R', 5: 'F-P', 6: 'UnF-P', 7: 'Mac-P', 8: 'F-F1', 9: 'UnF-F1',
                  10: 'Mac-F1', 11: 'AUC', 12: 'MCC', 13: 'SPD', 14: 'AOD', 15: 'EOD', 16: 'ERD'}

for j in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    for name in ['fairsmote', 'eo',  'ceo1', 'ceo2', 'ceo3', 'default', 'dir', 'lfr', 'rw', 'roc1', 'roc2', 'roc3']:
        for dataset in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age', 'Bank-Age', 'Mep-Race']:
            (dataset_pre, dataset_aft) = dataset.lower().split('-')
            if dataset == 'Mep-Race':
                dataset_pre = 'mep'
                dataset_aft = 'RACE'
            fin = open('../Results/' + name + '_' + j + '_' + dataset_pre + '_' + dataset_aft + '.txt', 'r')
            count = 0
            for line in fin:
                count = count + 1
                data[j][dataset][data_key_value[count]][name] = list(map(float, line.strip().split('\t')[1:51]))
            fin.close()

    for name in ['fairway', 'ad', 'pr', 'mfc1', 'mfc2']:
        for dataset in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age', 'Bank-Age', 'Mep-Race']:
            (dataset_pre, dataset_aft) = dataset.lower().split('-')
            if dataset == 'Mep-Race':
                dataset_pre = 'mep'
                dataset_aft = 'RACE'
            fin = open('../Results/' + name + '_lr_' + dataset_pre + '_' + dataset_aft + '.txt', 'r')
            count = 0
            for line in fin:
                count = count + 1
                data[j][dataset][data_key_value[count]][name] = list(map(float, line.strip().split('\t')[1:51]))
            fin.close()

    for name in ['op']:
        for dataset in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age']:
            (dataset_pre, dataset_aft) = dataset.lower().split('-')
            fin = open('../Results/' + name + '_' + j + '_' + dataset_pre + '_' + dataset_aft + '.txt', 'r')
            count = 0
            for line in fin:
                count = count + 1
                data[j][dataset][data_key_value[count]][name] = list(map(float, line.strip().split('\t')[1:51]))
            fin.close()
        for mm in ['F-P', 'F-R', 'F-F1', 'UnF-P', 'UnF-R', 'UnF-F1', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC',
                  'SPD', 'AOD', 'EOD', 'ERD']:
            data[j]['Bank-Age'][mm][name] = '/'
            data[j]['Mep-Race'][mm][name] = '/'

region_count = {}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1', 'AUC','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo', 'fairway', 'fairsmote']:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        for fairmetric in ['SPD','AOD','EOD']:
            for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','AUC','MCC']:
                for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo', 'fairway', 'fairsmote']:
                    if name == 'op' and j in ['Bank-Age','Mep-Race']:
                        continue
                    methods = dict()
                    name_fair50 = data[i][j][fairmetric][name]
                    name_per50 = data[i][j][permetric][name]
                    for count in range(50):
                        methods[str(count)] = (float(name_per50[count]), float(name_fair50[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1


name_real = {'op': 'OP', 'lfr': 'LFR', 'rw': 'RW', 'dir': 'DIR', 'pr': 'PR', 'ad': 'AD', 'mfc1': 'MFC-FDR',
             'mfc2': 'MFC-SR', 'roc1': 'ROC-SPD', 'roc2': 'ROC-AOD', 'roc3': 'ROC-EOD', 'ceo1': 'CEO-FNR',
             'ceo2': 'CEO-FPR', 'ceo3': 'CEO-W', 'eo': 'EOP', 'fairway': 'Fairway', 'fairsmote': 'Fair-SMOTE'}

fout = open('fig8_result','w')
fout.write('Results for Figure 8(a)----------------------------\n')
for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo', 'fairway', 'fairsmote']:
    fout.write(name_real[name])
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']:
                for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex','German-Age', 'Bank-Age', 'Mep-Race']:
                    if name == 'op' and j in ['Bank-Age','Mep-Race']:
                        continue
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    final_sum = 0
    for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')

fout.write('Results for Figure 8(b)----------------------------\n')
for i in ['lr', 'rf', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    fout.write(i)
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2','ceo3', 'eo', 'fairway', 'fairsmote']:
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']:
                for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex','German-Age', 'Bank-Age', 'Mep-Race']:
                    if name == 'op' and j in ['Bank-Age','Mep-Race']:
                        continue
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    final_sum = 0
    for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')

fout.write('Results for Figure 8(c)----------------------------\n')
for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex','German-Age', 'Bank-Age', 'Mep-Race']:
    fout.write(j)
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2','ceo3', 'eo', 'fairway', 'fairsmote']:
        for fairmetric in ['SPD', 'AOD', 'EOD']:
            for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']:
                for i in ['lr', 'rf', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
                    if name == 'op' and j in ['Bank-Age','Mep-Race']:
                        continue
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    final_sum = 0
    for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')

fout.write('Results for Figure 8(d)----------------------------\n')
for fairmetric in ['SPD', 'AOD', 'EOD']:
    for permetric in ['Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']:
        fout.write(fairmetric+'&'+permetric)
        final_count = {}
        for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
            final_count[region_kind] = 0
        for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2','ceo3', 'eo', 'fairway', 'fairsmote']:
            for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex','German-Age', 'Bank-Age', 'Mep-Race']:
                for i in ['lr', 'rf', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
                    if name == 'op' and j in ['Bank-Age','Mep-Race']:
                        continue
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
        final_sum = 0
        for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
            final_sum += final_count[region_kind]
        for region_kind in ['lose-lose', 'inverted', 'bad', 'good', 'win-win']:
            fout.write('\t%f' % (final_count[region_kind]/final_sum))
        fout.write('\n')

fout.close()