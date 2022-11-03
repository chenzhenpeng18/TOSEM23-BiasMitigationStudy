from numpy import mean, std, sqrt
import scipy.stats as stats


def cohen_d(x, y):
    return abs(mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)


def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]


data = {}
for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    data[i] = {}
    for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age', 'Bank-Age', 'Mep-Race']:
        data[i][j] = {}
        for k in ['F-P', 'F-R', 'F-F1', 'UnF-P', 'UnF-R', 'UnF-F1', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC',
                  'SPD', 'AOD', 'EOD', 'ERD']:
            data[i][j][k] = {}

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

metric_list = ['F-P', 'F-R', 'F-F1', 'UnF-P', 'UnF-R', 'UnF-F1', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'AUC', 'MCC']

diff_degree = {}
for i in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo',
          'fairway', 'fairsmote']:
    diff_degree[i] = {}
    for j in ['noorincrease', 'small', 'medium', 'large']:
        diff_degree[i][j] = 0

for z in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age', 'Bank-Age', 'Mep-Race']:
    for k in metric_list:
        for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
            default_list = data[algo][z][k]['default']
            default_valuefork = mean(default_list)
            for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
                # print(algo, z, k, name)
                if data[algo][z][k][name] != '/':
                    real_list = data[algo][z][k][name]
                    real_valuefork = mean(real_list)
                    rise_ratio = real_valuefork - default_valuefork
                    cohenn = cohen_d(default_list, real_list)
                    if mann(default_list, real_list) >= 0.05 or rise_ratio >= 0:
                        diff_degree[name]['noorincrease'] += 1
                    elif cohenn < 0.5:
                        diff_degree[name]['small'] += 1
                    elif cohenn >= 0.5 and cohenn < 0.8:
                        diff_degree[name]['medium'] += 1
                    elif cohenn >= 0.8:
                        diff_degree[name]['large'] += 1

name_real = {'op': 'OP', 'lfr': 'LFR', 'rw': 'RW', 'dir': 'DIR', 'pr': 'PR', 'ad': 'AD', 'mfc1': 'MFC-FDR',
             'mfc2': 'MFC-SR', 'roc1': 'ROC-SPD', 'roc2': 'ROC-AOD', 'roc3': 'ROC-EOD', 'ceo1': 'CEO-FNR',
             'ceo2': 'CEO-FPR', 'ceo3': 'CEO-W', 'eo': 'EOP', 'fairway': 'Fairway', 'fairsmote': 'Fair-SMOTE'}

fout = open("fig4_result", 'w')
for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo', 'fairway', 'fairsmote']:
    fout.write(name_real[name])
    summ = diff_degree[name]['noorincrease'] + diff_degree[name]['small'] + diff_degree[name]['medium'] + \
           diff_degree[name]['large']
    for j in ['noorincrease', 'small', 'medium', 'large']:
        fout.write('\t%f' % (diff_degree[name][j] / summ))
    fout.write('\n')
fout.close()