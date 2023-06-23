import torch
import os
import glob
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


class DataSet:
	def __init__(self, path_to_data="../data/full_dataset/goemotions_1.csv"):
	#def __init__(self, path_to_data="data/reduced_dataset/goemotions_small.csv"):
		# csv_data_list = glob.glob(path_to_data + '*.csv')
		# self.dataframes = []
		# for path in csv_data_list:
		# 	print(path)
		# 	df = pd.read_csv(path)
		# 	self.dataframes.append(df)
		# self.data = pd.concat(self.dataframes)
		self.data = pd.read_csv(path_to_data)
		self.labels = None
		self.texts = None
		self.tokenized_inputs = None
		# self.attention_mask = None
		self.max_text_len = None
		self.vocab_size = None
		self.duplicates = None
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
		self.num_classes = len(self.emotions_dict)

	def preprocessing_data(self, generate_from_scratch=False, data_augmentation=False, force_equality=False):
		if generate_from_scratch and data_augmentation:
			raise ValueError("Augmented Data created before using GPT prompt, get it from full_data")
		if data_augmentation:
			data_file_path = "../data/augmented_data/augmented_data_complete.csv"
		else:
			data_file_path = "../data/full_dataset/origin_data_proccessed.csv"

		if generate_from_scratch:
			self.remove_unclear_samples()
			self.add_emotion_label()
		else:
			self.data = pd.read_csv(data_file_path)
			self.data = self.data.sample(frac=1).reset_index(drop=True)
			self.labels = self.data['Emotion'].values.tolist()
			self.texts = self.data['text'].values.tolist()
			for label in self.labels:
				self.emotions_dict[self.generic_emotions_list[label]]["samples_num"] += 1

		if force_equality:
			min_samples = min(self.count_labels(to_stdout=False))

	def remove_unclear_samples(self):
		# Remove all the unclear text from the data and the duplicates
		self.data = self.data[self.data["example_very_unclear"] != True]

	def remove_duplicates(self):
		self.duplicates = self.data[self.data.duplicated(subset=['text'])]
		self.data = self.data.drop_duplicates(subset=['text'])

	def add_emotion_label(self):
		emotions_list = []
		emotions_start_index = 8
		# For each text, get the relevant emotion. If there are more than one, choose the first one
		# ########### Need to explain it on presentation or compare to duplicate scenario
		for index, row in self.data.iterrows():
			columns_with_value_one = self.data.columns[emotions_start_index:][row[emotions_start_index:] == 1].tolist()
			# create a duplication case afterwards
			specific_emotion = columns_with_value_one[0]
			for generic_emotion in self.emotions_dict:
				if specific_emotion in self.emotions_dict[generic_emotion]["semantics_feelings"]:
					emotions_list.append(generic_emotion)
					self.emotions_dict[generic_emotion]["samples_num"] += 1
					break
		# Save labels inside the data
		labels_list = [self.emotions_dict[emotion]["label"] for emotion in emotions_list]
		self.data['Emotion'] = labels_list
		# Save the labels list and texts as tensors
		self.labels = labels_list
		self.texts = self.data['text'].values.tolist()

	def count_labels(self, to_stdout=True):
		num_samples_list = []
		for generic_emotion in self.emotions_dict:
			num_samples = self.emotions_dict[generic_emotion]["samples_num"]
			num_samples_list.append(num_samples)
			if to_stdout:
				print(f"number of samples for {generic_emotion} is {num_samples}")
		return num_samples_list

	def tokenizer(self):
		if os.path.exists("./tokenizer"):
			try:
				tokenizer = BertTokenizer.from_pretrained("./tokenizer")
				print("Loaded tokenizer from directory.")

			except:
				print("Could not load tokenizer from directory. Training new tokenizer...")
				tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
				tokenizer.save_pretrained("./tokenizer")

		else:
			print("Directory does not exist. Training new tokenizer...")
			tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
			tokenizer.save_pretrained("./tokenizer")
		# self.tokenized_inputs = tokenizer(self.texts, padding=True, truncation=True, max_length=self.get_max_text_length())
		self.tokenized_inputs = tokenizer.batch_encode_plus(self.texts, add_special_tokens=True,
															return_attention_mask=True, pad_to_max_length=True,
															truncation=True,
															max_length=min(self.get_max_text_length(), 512),
															return_tensors='pt')
		self.vocab_size = tokenizer.vocab_size

	def split_train_test_val_data(self, test_size=0.15, val_size=0.15):
		test_val_size = test_size + val_size
		dataset = TensorDataset(torch.tensor(self.tokenized_inputs['input_ids']), 
                            	torch.tensor(self.tokenized_inputs['attention_mask']), 
                            	torch.tensor(self.labels))
		train_dataset, temp_dataset = train_test_split(dataset, test_size=test_val_size, random_state=42)
		test_from_val_size = test_size / (test_size + val_size)
		val_dataset, test_dataset = train_test_split(temp_dataset, test_size=test_from_val_size, random_state=42)
		return train_dataset, val_dataset, test_dataset

	def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size):
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
		return train_loader, test_loader, val_loader

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

