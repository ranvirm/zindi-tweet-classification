import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from datetime import datetime
import logging as log

log.basicConfig(
	format='%(levelname)s-%(asctime)s - %(message)s',
	datefmt='%d-%b-%y %H:%M:%S',
	level=log.DEBUG
)

# main_train_data = pd.read_csv('data/sources/train-2.csv')[:10]
# main_test_data = pd.read_csv('data/sources/test-2.csv')

bert_model = None
tokenizer = None


def create_bert_features(data_frame, text_col):
	# distilBERT
	# model_class, tokenizer_class, pretrained_weights = (
	# 	ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased'
	# )

	# BERT
	model_class, tokenizer_class, pretrained_weights = (
	ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased'
	)

	if globals()['tokenizer'] is not None:
		log.info('Tokenizer model already initialised.')
	else:
		# Load pre-trained model/tokenizer
		log.info('Initialising tokenizer.')
		global tokenizer
		tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
		log.info('Tokenizer initialised.')

	if globals()['bert_model'] is not None:
		log.info('BERT model already initialised.')
	else:
		log.info('Initialising BERT model.')
		global bert_model
		bert_model = model_class.from_pretrained(pretrained_weights)
		log.info('Initialised BERT model')

	tokenized = data_frame[text_col].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

	max_len = 0
	for i in tokenized.values:
		if len(i) > max_len:
			max_len = len(i)

	padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

	# input_ids = torch.tensor(np.array(padded))
	#
	# with torch.no_grad():
	# 	last_hidden_states = model(input_ids)

	attention_mask = np.where(padded != 0, 1, 0)

	input_ids = torch.tensor(padded)

	attention_mask = torch.tensor(attention_mask)

	with torch.no_grad():
		last_hidden_states = bert_model(input_ids, attention_mask=attention_mask)

	features = last_hidden_states[0][:, 0, :].numpy()

	return features


def create_bert_features_batcher(data_frame, text_col, batch_size=1000):
	assert type(data_frame) == pd.DataFrame
	n_rows = data_frame.shape[0]
	log.info(f'Number of batches: {round(n_rows/batch_size, 0)}')
	ff = []
	for i in range(0, n_rows, batch_size-1):
		log.info(f'Started extraction of BERT feautures for batch rows {i} to {i+batch_size-1}')
		f = create_bert_features(data_frame=data_frame[i:i+batch_size-1], text_col=text_col)
		log.info(f'Completed extraction of BERT main_train_features for batch rows {i} to {i+batch_size-1}')
		ff.append(f)
	return np.concatenate(ff)

#
# # +++
# # train model
# # main_train_features = create_bert_features(data_frame=main_train_data[:400], text_col='text')
# main_train_features = create_bert_features_batcher(data_frame=main_train_data, text_col='text', batch_size=100)
# labels = main_train_data['target']
#
# train_features, test_features, train_labels, test_labels = train_test_split(main_train_features, labels)
#
# # grid search
# parameters = [
# 	{
# 		'C': np.linspace(0.001, 50, 20),
# 		'fit_intercept': [True, False],
# 		'verbose': [0],
# 		'solver':('newton-cg', 'sag', 'saga')
# 	},
# 	{
# 		'C': np.linspace(0.001, 50, 20),
# 		'fit_intercept': [True, False],
# 		'verbose': [1],
# 		'solver': ('lbfgs', 'liblinear')
# 	}
#
# ]
# grid_search = GridSearchCV(LogisticRegression(max_iter=400), parameters)
# grid_search.fit(train_features, train_labels)
#
# log.info(f'Best parameters: \n { grid_search.best_params_}')
# log.info(f'Best scrores: \n {grid_search.best_score_}')
#
# # train using best params
# lr_clf = LogisticRegression(
# 	C=grid_search.best_params_['C'],
# 	fit_intercept=grid_search.best_params_['fit_intercept'],
# 	solver=grid_search.best_params_['solver'],
# 	max_iter=100
# )
# lr_clf.fit(train_features, train_labels)
#
# # save model
# pickle.dump(lr_clf, open(f'''models/logistic/older/{datetime.now().strftime('%d-%b-%y_%H:%M:%S')}.pkl''', 'wb'))
# pickle.dump(lr_clf, open(f'''models/logistic/latest/latest.pkl''', 'wb'))
#
# # label test data
# y_pred_binary = lr_clf.predict(test_features)
# y_pred_prob = lr_clf.predict_proba(test_features)
#
# # compute metrics
# score = lr_clf.score(test_features, test_labels)
# cfm = metrics.confusion_matrix(test_labels, y_pred_binary)
# precision = metrics.precision_score(test_labels, y_pred_binary)
# log_loss = metrics.log_loss(test_labels.astype('float'), y_pred_prob)
#
# print(f'Accuracy: {score}')
# print(f'Confusion Matrix: {cfm}')
# print(f'Precision: {precision}')
# print(f'Log Loss: {log_loss}')
#
# # prepare test data
# main_test_features = create_bert_features_batcher(data_frame=main_test_data, text_col='text', batch_size=100)
#
# # classify test data
# main_test_labelled = lr_clf.predict_proba(main_test_features)
# main_test_labelled_df = pd.DataFrame(main_test_labelled)
# main_test_data['target'] = main_test_labelled_df[[1]]
# main_test_predictions = main_test_data[['ID', 'target']]
# main_test_predictions.to_csv(f'''data/output/older/{datetime.now().strftime('%d-%b-%y_%H:%M:%S')}.csv''', index=False)
# main_test_predictions.to_csv(f'''data/output/latest/latest.csv''', index=False)
#
# # +++
# # test predictions
# # test_features = create_bert_features(data_frame=main_test_data, text_col='text')
# # test_labels = lr_clf.predict_proba(test_features)
