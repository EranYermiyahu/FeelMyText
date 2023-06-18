import torch
import glob
import pandas as pd


class DataSet:
	def __init__(self, path_to_data="../data/full_dataset/goemotions_1.csv"):
		path_to_data = path_to_data
		# csv_data_list = glob.glob(path_to_data + '*.csv')
		# self.dataframes = []
		# for path in csv_data_list:
		# 	print(path)
		# 	df = pd.read_csv(path)
		# 	self.dataframes.append(df)
		# self.data = pd.concat(self.dataframes)
		self.data = pd.read_csv(path_to_data)
		self.duplicates = None
		self.emotions_dict = {
			"anger": {
				"semantics_feelings": ["anger", "annoyance", "disapproval"],
				"label": 0
			},
			"fear": {
				"semantics_feelings": ["fear", "nervousness"],
				"label": 1
			},
			"joy": {
				"semantics_feelings": ["joy", "amusement", "approval", "gratitude", "optimism", "relief",
									   "pride", "admiration"],
				"label": 2
			},
			"passion": {
				"semantics_feelings": ["excitement", "love", "caring", "desire"],
				"label": 3
			},
			"sadness": {
				"semantics_feelings": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
				"label": 4
			},
			"surprise": {
				"semantics_feelings": ["surprise", "realization", "confusion", "curiosity"],
				"label": 5
			},
			"neutral":{
				"semantics_feelings": ["neutral"],
				"label": 6
			}
		}

	def remove_unclear_samples(self):
		# Remove all the unclear text from the data and the duplicates
		self.data = self.data[self.data["example_very_unclear"] != True]

	def remove_duplicates(self):
		self.duplicates = self.data[self.data.duplicated(subset=['text'])]
		self.data = self.data.drop_duplicates(subset=['text'])

	def add_emotion_label(self):
		pass

	def print_lines(self):
		print(self.data.shape[0])
