import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from mdsp import MDSP

class ALCE:
	def __init__(self, x_pool, y_pool, cost_mat, base=SVR):
		self.x_pool = x_pool
		self.y_pool = y_pool
		self.C = cost_mat.shape[0]
		self.base = [base() for c in xrange(self.C)]
		self.cost_mat = cost_mat
		self.dis_mat = np.zeros((2*self.C, 2*self.C))
		self.dis_mat[:self.C, self.C:] = cost_mat
		self.dis_mat[self.C:, :self.C] = cost_mat.T
		self.mds = MDSP(metric=False, n_components=self.C, n_uq=self.C, max_iter=300, eps=1e-6, dissimilarity="precomputed", n_init=1, n_jobs=1)
		self.y_embedding = self.mds.fit(self.dis_mat).embedding_
		self.nn_embedding = NearestNeighbors(n_neighbors=1)
		self.nn_embedding.fit(self.y_embedding[self.C:, :])
		y_train = self.y_embedding[self.y_pool[self.y_pool>0]]
		for c in xrange(self.C):
			self.base[c].fit(self.x_pool[self.y_pool>0], y_train[:, c])

	def query(self):
		p_embedding = np.zeros((np.sum(self.y_pool==0), self.C))
		for c in xrange(self.C):
			p_embedding[:, c] = self.base[c].predict(self.x_pool[self.y_pool==0])
		uncertainty, pred = self.nn_embedding.kneighbors(p_embedding)
		return np.where(self.y_pool==0)[0][uncertainty.argmax()]

	def update(self, idx, y):
		self.y_pool[idx] = y
		y_train = self.y_embedding[self.y_pool[self.y_pool>0]]
		for c in xrange(self.C):
			self.base[c].fit(self.x_pool[self.y_pool>0], y_train[:, c])
