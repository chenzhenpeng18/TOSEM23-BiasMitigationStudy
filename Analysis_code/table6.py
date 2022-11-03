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


region_area = {}
for dataset in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    region_area[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_area[dataset][fairmetric] = {}
        for permetric in ['AUC', 'Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_area[dataset][fairmetric][permetric]={}
            for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
                region_area[dataset][fairmetric][permetric][algo]={}


for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    for j in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
        for fairmetric in ['SPD','AOD','EOD']:
            for permetric in ['AUC', 'Acc','Mac-P','Mac-R','Mac-F1','MCC']:
                methods = dict()
                methods_list = ['lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']
                if j not in ['Bank-Age','Mep-Race']:
                    methods_list.append('op')
                for name in methods_list:
                    name_fair50 = list(map(float,data[i][j][fairmetric][name]))
                    name_per50 = list(map(float,data[i][j][permetric][name]))
                    methods[name]=(mean(name_per50),mean(name_fair50))
                normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                mitigation_regions = classify_region(baseline, normalized_methods)
                for name in mitigation_regions:
                    region_area[j][fairmetric][permetric][i][name] = mitigation_regions[name]
                good = {k for k, v in mitigation_regions.items() if v == "good"}
                normalized_methods = {k: v for k, v in normalized_methods.items() if k in good}
                for k, v in normalized_methods.items():
                    area = compute_area(baseline, v)
                    region_area[j][fairmetric][permetric][i][k]=area


prop_count = {}
fout = open('table6_result','w')
for z in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    fout.write(z)
    prop_count[z] = {}
    for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
        prop_count[z][name]=0
    for permetric in ['AUC','Acc','Mac-P','Mac-R','Mac-F1','MCC']:
        for fairmetric in ['SPD','AOD','EOD']:
            for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
                for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
                    if name == 'op' and z in ['Bank-Age','Mep-Race']:
                        continue
                    if region_area[z][fairmetric][permetric][i][name] == 'win-win':
                        region_area[z][fairmetric][permetric][i][name] = 100
                    elif region_area[z][fairmetric][permetric][i][name] == 'lose-lose':
                        region_area[z][fairmetric][permetric][i][name] = -100
                    elif region_area[z][fairmetric][permetric][i][name] == 'bad':
                        region_area[z][fairmetric][permetric][i][name] = -25
                    elif region_area[z][fairmetric][permetric][i][name] == 'inverted':
                        region_area[z][fairmetric][permetric][i][name] = -50
                    elif region_area[z][fairmetric][permetric][i][name] == 'good':
                        region_area[z][fairmetric][permetric][i][name] = 0
                max_method = []
                max_num = -1000
                for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
                    if name == 'op' and z in ['Bank-Age','Mep-Race']:
                        continue
                    if region_area[z][fairmetric][permetric][i][name] > max_num:
                        max_num = region_area[z][fairmetric][permetric][i][name]
                        max_method = [name]
                    elif region_area[z][fairmetric][permetric][i][name] == max_num:
                        max_method.append(name)
                for name in max_method:
                    prop_count[z][name] = prop_count[z][name]+1
    for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
        if name == 'op' and z in ['Bank-Age', 'Mep-Race']:
            fout.write(' & -')
            continue
        fout.write(' & %d' % int(100*prop_count[z][name]/126))
        fout.write('\\%')
    fout.write('\\\\')
    fout.write('\n')

fout.write('Overall')
for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
    total_count = 0
    for z in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age', 'Bank-Age',
              'Mep-Race']:
        if name == 'op' and z in ['Bank-Age', 'Mep-Race']:
            continue
        total_count += prop_count[z][name]
    if name == 'op':
        fout.write(' & %d' % int(100*total_count/756))
        fout.write('\\%')
    else:
        fout.write(' & %d' % int(100 * total_count /1008))
        fout.write('\\%')
fout.write('\\\\')
fout.write('\n')

fout.close()

