import torch
import os
import glob
import random
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


class DataSet:
	def __init__(self, tokenizer, path_to_data="../data/full_dataset/goemotions_1.csv"):
	#def __init__(self, path_to_data="data/reduced_dataset/goemotions_small.csv"):
		# csv_data_list = glob.glob(path_to_data + '*.csv')
		# self.dataframes = []
		# for path in csv_data_list:
		# 	print(path)
		# 	df = pd.read_csv(path)
		# 	self.dataframes.append(df)
		# self.data = pd.concat(self.dataframes)
		self.data = pd.read_csv(path_to_data)
		self.tokenizer = tokenizer
		self.labels = None
		self.texts = None
		self.tokenized_inputs = None
		# self.attention_mask = None
		self.max_text_len = None
		self.vocab_size = self.tokenizer.vocab_size
		self.duplicates = None
		self.generalize_emotions_flag = True
		self.num_classes = None
		self.class_columns = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
		self.generic_emotions_list = ['anger', 'revulsion', 'joy', 'passion', 'sadness', 'surprise', 'neutral']
		self.emotions_dict = {
			"anger": {
				"semantics_feelings": ["anger", "annoyance"],
				"label": 0,
				"samples_num": 0
			},
			"revulsion": {
				"semantics_feelings": ["fear", "nervousness", "disgust", "disapproval"],
				"label": 1,
				"samples_num": 0
			},
			"joy": {
				"semantics_feelings": ["joy", "amusement", "approval", "gratitude", "optimism", "relief", "pride", "admiration"],
				"label": 2,
				"samples_num": 0
			},
			"passion": {
				"semantics_feelings": ["excitement", "love", "caring", "desire"],
				"label": 3,
				"samples_num": 0
			},
			"sadness": {
				"semantics_feelings": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
				"label": 4,
				"samples_num": 0
			},
			"surprise": {
				"semantics_feelings": ["surprise", "realization", "confusion", "curiosity"],
				"label": 5,
				"samples_num": 0
			},
			"neutral": {
				"semantics_feelings": ["neutral"],
				"label": 6,
				"samples_num": 0
			}
		}

	# def preprocessing_data(self, generate_from_scratch=False, data_augmentation=False, force_equality=False):
	# 	if generate_from_scratch and data_augmentation:
	# 		raise ValueError("Augmented Data created before using GPT prompt, get it from full_data")
	# 	if data_augmentation:
	# 		data_file_path = "../data/augmented_data/augmented_data_complete.csv"
	# 	else:
	# 		data_file_path = "../data/full_dataset/raw_emotions_data.csv"
	#
	# 	if generate_from_scratch:
	# 		self.remove_unclear_samples()
	# 		self.add_emotion_label(generalize_emotions=False)
	# 	else:
	# 		self.data = pd.read_csv(data_file_path)
	# 		self.data = self.data.sample(frac=1).reset_index(drop=True)
	# 		self.labels = self.data['Emotion'].values.tolist()
	# 		self.texts = self.data['text'].values.tolist()
	# 		self.num_classes = len(self.class_columns) if self.generalize_emotions_flag else len(
	# 			self.generic_emotions_list)
	# 		for label in self.labels:
	# 			self.emotions_dict[self.generic_emotions_list[label]]["samples_num"] += 1
	#
	# 	if force_equality:
	# 		min_samples = min(self.count_labels(to_stdout=False))

	def preprocessing_data(self, data_augmentation=False):
		if data_augmentation:
			data_file_path = "../data/augmented_data/augmented_data_complete.csv"
		else:
			data_file_path = "../data/full_dataset/raw_emotions_data.csv"

		self.data = pd.read_csv(data_file_path)
		self.data = self.data.sample(frac=1).reset_index(drop=True)
		self.labels = self.data['Emotion'].values.tolist()
		self.texts = self.data['text'].values.tolist()
		self.num_classes = len(self.generic_emotions_list) if self.generalize_emotions_flag else len(self.class_columns)
		if self.generalize_emotions_flag:
			for label in self.labels:
				self.emotions_dict[self.generic_emotions_list[label]]["samples_num"] += 1

	def remove_unclear_samples(self):
		# Remove all the unclear text from the data and the duplicates
		self.data = self.data[self.data["example_very_unclear"] != True]

	def remove_duplicates(self):
		self.duplicates = self.data[self.data.duplicated(subset=['text'])]
		self.data = self.data.drop_duplicates(subset=['text'])

	def add_emotion_label(self, generalize_emotions=True):
		self.generalize_emotions_flag = generalize_emotions
		emotions_list = []
		emotions_start_index = 8
		# For each text, get the relevant emotion. If there are more than one, choose the first one
		# ########### Need to explain it on presentation or compare to duplicate scenario
		for index, row in self.data.iterrows():
			columns_with_value_one = self.data.columns[emotions_start_index:][row[emotions_start_index:] == 1].tolist()
			# create a duplication case afterwards
			specific_emotion = columns_with_value_one[0]
			if generalize_emotions:
				for generic_emotion in self.emotions_dict:
					if specific_emotion in self.emotions_dict[generic_emotion]["semantics_feelings"]:
						generic_emotion_label = self.emotions_dict[generic_emotion]["label"]
						emotions_list.append(generic_emotion_label)
						self.emotions_dict[generic_emotion]["samples_num"] += 1
						break
			else:
				emotions_list.append(self.data.columns.get_loc(specific_emotion) - (emotions_start_index + 1))
		self.data['Emotion'] = emotions_list
		# Save the labels list and texts as tensors
		self.labels = emotions_list
		self.texts = self.data['text'].values.tolist()
		self.num_classes = len(self.generic_emotions_list) if self.generalize_emotions_flag else len(self.class_columns)

	def count_labels(self, to_stdout=True):
		if self.generalize_emotions_flag:
			num_samples_list = []
			for generic_emotion in self.emotions_dict:
				num_samples = self.emotions_dict[generic_emotion]["samples_num"]
				num_samples_list.append(num_samples)
				if to_stdout:
					print(f"number of samples for {generic_emotion} is {num_samples}")
		else:
			num_samples_list = [0] * (max(self.labels) + 1)
			for lbl in self.labels:
				num_samples_list[lbl] += 1
			if to_stdout:
				for lbl, lbl_rep in enumerate(num_samples_list):
					print(f"number of samples for {self.class_columns[lbl]} is {lbl_rep}")
		return num_samples_list

	def tokenizer(self):
		self.tokenized_inputs = self.tokenizer.batch_encode_plus(self.texts, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, truncation=True, max_length=512, return_tensors='pt')

	def split_train_val_data(self, val_size=0.15):
		dataset = TensorDataset(torch.tensor(self.tokenized_inputs['input_ids']),
                            	torch.tensor(self.tokenized_inputs['attention_mask']), 
                            	torch.tensor(self.labels))
		train_dataset, val_dataset = train_test_split(dataset, test_size=val_size, random_state=42)
		return train_dataset, val_dataset

	def create_data_loaders(self, train_dataset, val_dataset, batch_size):
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
		return train_loader, val_loader

	def get_max_text_length(self):
		if self.max_text_len is not None:
			return self.max_text_len
		self.max_text_len = max(len(sentence) for sentence in self.texts)
		return self.max_text_len

	def get_class_counts(self):
		self.class_counts = self.data[self.class_columns].sum()
		print(self.class_counts)
		return self.class_counts

	def print_lines(self):
		print(self.data.shape[0])




