import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from query_strategies.uncertainty_margin import UncertaintyMargin
from query_strategies.uncertainty_entropy import UncertaintyEntropy
from query_strategies.active_learning_with_cost_embedding import ALCE
from query_strategies.maximum_expected_cost import MEC
from query_strategies.cost_weighted_minimum_margin import CWMM
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import sys

# set random seed
np.random.seed(1)

# load 
x_data = np.loadtxt('vehicle/data.x', dtype=float)
y_data = np.loadtxt('vehicle/data.y', dtype=int)

C = 4 # number of class
N = x_data.shape[0] # number of instances
T = 400 # number of testing instances
Q = 200 # number of queries

def shuffle_data(x_data, y_data, N, T, C):
	idx = np.arange(N)
	np.random.shuffle(idx)
	x_pool = x_data[idx[:-T]]
	y_pool = y_data[idx[:-T]]
	x_test = x_data[idx[-T:]]
	y_test = y_data[idx[-T:]]
	N_train = x_pool.shape[0]

	# sample initial labeled instances
	idx_lbl = np.zeros((N_train, ), dtype=bool)
	for i in xrange(C):
		idx_lbl[np.random.choice(np.where(y_pool==(i+1))[0])] = True
	
	# generate cost matrix
	unique, counts = np.unique(y_data, return_counts=True)
	class_counts = dict(zip(unique, counts))
	cost_mat = np.zeros((C, C))
	for i in xrange(C):
		for j in xrange(C):
			if i == j:
				continue
			cost_mat[i, j] = np.random.random()*2000*class_counts[j+1]/class_counts[i+1]

	return x_pool, y_pool, x_test, y_test, idx_lbl, cost_mat

total_results = np.zeros((Q, 5))

# run several experiments
sys.stderr.write('total  #####\n')
sys.stderr.write('runing ')
for i in xrange(5):
	# shuffle dataset
	x_pool, y_pool, x_test, y_test, idx_lbl, cost_mat = shuffle_data(x_data, y_data, N, T, C)

	# different models
	models = [UncertaintyMargin(x_pool, y_pool*idx_lbl), 
		UncertaintyEntropy(x_pool, y_pool*idx_lbl),
		ALCE(x_pool, y_pool*idx_lbl, cost_mat),
		CWMM(x_pool, y_pool*idx_lbl, cost_mat),
		MEC(x_pool, y_pool*idx_lbl, cost_mat)]

	# for recording rewards and actions
	results = np.zeros((Q, len(models)))
	idx_lbls = np.repeat(idx_lbl[:, None], len(models) ,axis=1)

	for nq in xrange(Q):
		for j, model in enumerate(models):
			q = model.query()
			model.update(q, y_pool[q])
			idx_lbls[q, j] = True

			# testing performance
			clf = OneVsRestClassifier(SVC())
			clf.fit(x_pool[idx_lbls[:, j]], y_pool[idx_lbls[:, j]])
			p_test = clf.predict(x_test)
			results[nq, j] = np.mean([cost_mat[y-1, p-1] for y, p in zip(y_test, p_test)])
	# ipdb.set_trace()
	# print 1
		

	total_results += results
	sys.stderr.write('#')

sys.stderr.write('\nPlease see result.png\n')
avg_results = total_results/5

# plot result
plt.figure()
show_x = np.arange(0, Q, 10)
plt.plot(show_x, avg_results[::10, 0], '-bs', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='UncertaintyMargin')
plt.plot(show_x, avg_results[::10, 1], '-cv', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='UncertaintyEntropy')
plt.plot(show_x, avg_results[::10, 2], '-ro', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='ALCE')
plt.plot(show_x, avg_results[::10, 3], '-y^', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='CWMM')
plt.plot(show_x, avg_results[::10, 4], '-gd', fillstyle='none', linewidth=1.5, markeredgewidth=1.5, label='MEC')
plt.xlabel('number of queries')
plt.ylabel('average costs')
plt.legend(loc='upper right')
plt.savefig('result.png')
