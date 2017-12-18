import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

class CWMM:
	def __init__(self, x_pool, y_pool, cost_mat, base=SVC):
		self.x_pool = x_pool
		self.y_pool = y_pool
		self.cost_mat = cost_mat
		self.base = OneVsRestClassifier(base(probability=True))
		self.base.fit(x_pool[y_pool>0], y_pool[y_pool>0])

	def query(self):
		prob = self.base.predict_proba(self.x_pool[self.y_pool==0])
		prob_sort = prob.argsort(axis=1)
		cost1 = np.sum(prob*self.cost_mat.T[prob_sort[:, -1]], axis=1)
		cost2 = np.sum(prob*self.cost_mat.T[prob_sort[:, -2]], axis=1)
		uncertainty = cost1 - cost2
		return np.where(self.y_pool==0)[0][uncertainty.argmax()]

	def update(self, idx, y):
		self.y_pool[idx] = y
		self.base.fit(self.x_pool[self.y_pool>0], self.y_pool[self.y_pool>0])
