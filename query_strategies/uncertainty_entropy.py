import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

class UncertaintyEntropy:
	def __init__(self, x_pool, y_pool, base=SVC):
		self.x_pool = x_pool
		self.y_pool = y_pool
		self.base = OneVsRestClassifier(base(probability=True))
		self.base.fit(x_pool[y_pool>0], y_pool[y_pool>0])

	def query(self):
		prob = self.base.predict_proba(self.x_pool[self.y_pool==0])
		uncertainty = np.sum(-prob*np.log(prob), axis=1)
		return np.where(self.y_pool==0)[0][uncertainty.argmax()]

	def update(self, idx, y):
		self.y_pool[idx] = y
		self.base.fit(self.x_pool[self.y_pool>0], self.y_pool[self.y_pool>0])
