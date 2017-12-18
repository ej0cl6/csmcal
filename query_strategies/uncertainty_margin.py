import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

class UncertaintyMargin:
	def __init__(self, x_pool, y_pool, base=SVC):
		self.x_pool = x_pool
		self.y_pool = y_pool
		self.base = OneVsRestClassifier(base())
		self.base.fit(x_pool[y_pool>0], y_pool[y_pool>0])

	def query(self):
		margin = self.base.decision_function(self.x_pool[self.y_pool==0])
		uncertainty = -margin.max(axis=1)
		return np.where(self.y_pool==0)[0][uncertainty.argmax()]

	def update(self, idx, y):
		self.y_pool[idx] = y
		self.base.fit(self.x_pool[self.y_pool>0], self.y_pool[self.y_pool>0])
