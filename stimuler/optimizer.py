# Author:  DINDIN Meryll
# Date:    06 August 2019
# Project: DreemHeadband

try: from stimuler.imports import *
except: from imports import *
# Challenger Package
from optimizers import Prototype, Bayesian

class DataLoader:

	def __init__(self, directory='../data/slow_waves'):

		# Train dataset
		lab = pd.read_csv('/'.join([directory, 'label.csv']), index_col=0)
		f_0 = pd.read_parquet('/'.join([directory, 'train_cmp.pq']))
		f_1 = pd.read_parquet('/'.join([directory, 'train_fea.pq'])).drop('label', axis=1)
		f_0.index = f_1.index
		# Use as attributes
		self.x_t = f_0.join(f_1, how='left')
		self.y_t = lab.values.ravel()

		# Test dataset
		f_0 = pd.read_parquet('/'.join([directory, 'train_cmp.pq']))
		f_1 = pd.read_parquet('/'.join([directory, 'train_fea.pq'])).drop('label', axis=1)
		f_0.index = f_1.index
		# Use as attributes
		self.x_v = f_0.join(f_1, how='left')

		# Memory efficiency
		del lab, f_0, f_1

		# Embedded methods
		self._preprocess()
		self._categorize()

	def _preprocess(self):

		# Initialize scaler
		sts = StandardScaler()
		# Apply to train and test
		x_t = sts.fit_transform(self.x_t)
		self.x_t = pd.DataFrame(x_t, columns=self.x_t.columns, index=self.x_t.index)
		x_v = sts.transform(self.x_v)
		self.x_v = pd.DataFrame(x_v, columns=self.x_v.columns, index=self.x_v.index)
		# Memory efficiency
		del sts, x_t, x_v

	def _categorize(self):

		# Reapply binary categorization
		self.x_t['sleep_stage'] = (self.x_t['sleep_stage'] < 0).astype('int')
		self.x_v['sleep_stage'] = (self.x_v['sleep_stage'] < 0).astype('int')
