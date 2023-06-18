import torch
import glob
import pandas as pd


class DataSet:
	def __init__(self, path_to_data="../data/full_dataset/"):
		path_to_data = path_to_data
		csv_data_list = glob.glob(path_to_data + '*.csv')
		self.dataframes = []
		for path in csv_data_list:
			print(path)
			df = pd.read_csv(path)
			self.dataframes.append(df)
		self.data = pd.concat(self.dataframes)
		duplicates = self.data.duplicated()
		print(self.data[duplicates])
		print(self.data.shape[1])
