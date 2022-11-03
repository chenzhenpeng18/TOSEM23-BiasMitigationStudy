from numpy import mean, std,sqrt
import scipy.stats as stats

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

metric_list = ['F-P','F-R','F-F1','UnF-P','UnF-R','UnF-F1','AUC','Acc','Mac-P','Mac-R','Mac-F1','MCC']
diff_list = {}
for i in metric_list:
    diff_list[i] = []

result_corr = {}
for z in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    result_corr[z]={}
    for k in metric_list:
        result_corr[z][k]={}
        for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
            result_corr[z][k][algo]=[]
            default_list = data[algo][z][k]['default']
            default_valuefork = mean(default_list)
            for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
                if data[algo][z][k][name] != '/':
                    real_list = data[algo][z][k][name]
                    real_valuefork = mean(real_list)
                    rise_ratio = real_valuefork-default_valuefork
                    diff_list[k].append(rise_ratio)
                    result_corr[z][k][algo].append(rise_ratio)

for z in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    for i in metric_list:
        for j in metric_list:
            result_corr[z][i + '+' + j]={}

for z in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
    for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
        for i in metric_list:
            for j in metric_list:
                result_corr[z][i+'+'+j][algo] = stats.spearmanr(result_corr[z][i][algo],result_corr[z][j][algo])

fout = open("table2_result",'w')
i_count = 0
for i in metric_list:
    i_count+=1
    fout.write(i)
    j_count=0
    for j in metric_list:
        j_count+=1
        if j_count == 1:
            continue
        if j_count<=i_count:
            fout.write('& -')
            continue
        aresult = stats.spearmanr(diff_list[i],diff_list[j])
        fout.write('& %s' % (format(round(aresult[0],3),'.3f')))
        if aresult[1] < 0.05:
            fout.write('*')
        count = 0
        total_count=0
        for z in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
            for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
                total_count+=1
                if result_corr[z][i+'+'+j][algo][0] * aresult[0] > 0 and (aresult[1] - 0.05)*(result_corr[z][i+'+'+j][algo][1]-0.05)>0:
                    count=count+1
        fout.write('(%d/%d)' % (count,total_count))
    fout.write('\\\\\n')
fout.close()
