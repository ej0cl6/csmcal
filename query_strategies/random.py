import numpy as np

class Random:
	def __init__(self, x_pool, y_pool):
		self.x_pool = x_pool
		self.y_pool = y_pool

	def query(self):
		return np.random.choice(np.where(self.y_pool==0)[0])

	def update(self, idx, y):
		self.y_pool[idx] = y
