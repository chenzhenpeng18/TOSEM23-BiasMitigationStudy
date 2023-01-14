from numpy import mean

data = {}
for i in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
    data[i] = {}
    for j in ['Adult-Sex', 'Adult-Race', 'Compas-Sex', 'Compas-Race', 'German-Sex', 'German-Age', 'Bank-Age', 'Mep-Race']:
        data[i][j] = {}
        for k in ['F-P', 'F-R', 'F-F1', 'UnF-P', 'UnF-R', 'UnF-F1', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC',
                  'SPD', 'AOD', 'EOD', 'ERD']:
            data[i][j][k] = {}

data_key_value = {1: 'Acc', 2: 'F-R', 3: 'UnF-R', 4: 'Mac-R', 5: 'F-P', 6: 'UnF-P', 7: 'Mac-P', 8: 'F-F1', 9: 'UnF-F1',
                  10: 'Mac-F1', 11: 'MCC', 12: 'SPD', 13: 'AOD', 14: 'EOD', 15: 'ERD'}

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
        for mm in ['F-P', 'F-R', 'F-F1', 'UnF-P', 'UnF-R', 'UnF-F1', 'Acc', 'Mac-P', 'Mac-R', 'Mac-F1', 'MCC',
                  'SPD', 'AOD', 'EOD', 'ERD']:
            data[j]['Bank-Age'][mm][name] = '/'
            data[j]['Mep-Race'][mm][name] = '/'

metric_list = ['F-P','F-R','F-F1','UnF-P','UnF-R','UnF-F1','Acc','Mac-P','Mac-R','Mac-F1','MCC']
rank_result = {}
for eachmetric in metric_list:
	rank_result[eachmetric]={}
	for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
		rank_result[eachmetric][algo] ={}
		for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
			rank_result[eachmetric][algo][name]={}
			rank_result[eachmetric][algo][name]=[]


for i in ['Adult-Sex','Adult-Race','Compas-Sex','Compas-Race','German-Sex','German-Age','Bank-Age','Mep-Race']:
	for j in metric_list:
		rank_tmp = {}
		for algo in ['rf', 'lr', 'svm', 'dl', 'dl2', 'dl3', 'dl4']:
			default_list = data[algo][i][j]['default']
			default_valuefork = mean(default_list)
			for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
				if data[algo][i][j][name] == '/':
					continue
				real_list = data[algo][i][j][name]
				real_valuefork = mean(real_list)
				rank_tmp[name] = real_valuefork
			a_tmp = sorted(rank_tmp.items(), key=lambda x: x[1], reverse=True)
			for rankcount in range(len(a_tmp)):
				rank_result[j][algo][a_tmp[rankcount][0]].append(rankcount+1)

fout  = open('table3_result','w')
for j in metric_list:
	fout.write(j)
	for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
		fout.write(' & ')
		fout.write('%d' % int(mean(rank_result[j]['lr'][name]+rank_result[j]['svm'][name]+rank_result[j]['rf'][name]+rank_result[j]['dl'][name]+rank_result[j]['dl2'][name]+rank_result[j]['dl3'][name]+rank_result[j]['dl4'][name])))
	fout.write('\\\\\n')
# fout.write('Overall')
# for name in ['op', 'lfr', 'rw', 'dir', 'pr', 'ad', 'mfc1', 'mfc2', 'roc1', 'roc2', 'roc3', 'ceo1', 'ceo2', 'ceo3', 'eo','fairway', 'fairsmote']:
# 	fout.write(' & ')
# 	ttmp = []
# 	for j in metric_list:
# 		ttmp = ttmp+rank_result[j]['lr'][name]+rank_result[j]['svm'][name]+rank_result[j]['rf'][name]+rank_result[j]['dl'][name]+rank_result[j]['dl2'][name]+rank_result[j]['dl3'][name]+rank_result[j]['dl4'][name]
# 	fout.write('%d' % int(mean(ttmp)))
# fout.write('\\\\\n')
fout.close()


