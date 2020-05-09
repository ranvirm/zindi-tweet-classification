from bert import create_bert_features_batcher
from helpers.entity_extraction import extract_entities

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
import os

log.basicConfig(
	format='%(levelname)s-%(asctime)s - %(message)s',
	datefmt='%d-%b-%y %H:%M:%S',
	level=log.INFO
)

# +++ SETUP
USE_PERSISTED_FEATURES = False
DO_CV = False

# load source data
main_train_data = pd.read_csv('data/sources/train-2.csv')[:None]
main_test_data = pd.read_csv('data/sources/test-2.csv')

# +++
if USE_PERSISTED_FEATURES:
	if os.path.isfile('data/sources/train_features.npy') and os.path.isfile('data/sources/test_features.npy'):
		log.info('Using persisted features.')
		main_train_features = np.load('data/sources/train_features.npy')
		main_test_features = np.load('data/sources/test_features.npy')
	else:
		USE_PERSISTED_FEATURES = False

if not USE_PERSISTED_FEATURES:
	# prepare train main_train_features
	# create bert main_train_features
	# main_train_features = create_bert_features(data_frame=main_train_data[:400], text_col='text')
	bert_features = create_bert_features_batcher(data_frame=main_train_data, text_col='text', batch_size=200)
	entity_features = extract_entities(data_frame=main_train_data, text_col='text')
	main_train_features = np.array([np.concatenate((bert_features[i], entity_features[i])) for i in range(0, len(bert_features))])
	np.save('data/sources/train_features.npy', main_train_features)

	# +++
	# prepare test main_train_features
	# create bert main_train_features
	# main_train_features = create_bert_features(data_frame=main_train_data[:400], text_col='text')
	bert_features = create_bert_features_batcher(data_frame=main_test_data, text_col='text', batch_size=200)
	entity_features = extract_entities(data_frame=main_test_data, text_col='text')
	main_test_features = np.array([np.concatenate((bert_features[i], entity_features[i])) for i in range(0, len(bert_features))])
	np.save('data/sources/test_features.npy', main_test_features)

# train labels
labels = main_train_data['target']

# split data for training
train_features, test_features, train_labels, test_labels = train_test_split(main_train_features, labels)

if DO_CV:
	# grid search
	parameters = [
		{
			'C': np.linspace(0.0001, 100, 20),
			'fit_intercept': [True, False],
			'verbose': [0],
			'solver':('newton-cg', 'sag', 'saga')
		},
		{
			'C': np.linspace(0.0001, 100, 20),
			'fit_intercept': [True, False],
			'verbose': [1],
			'solver': ('lbfgs', 'liblinear')
		}

	]
	grid_search = GridSearchCV(LogisticRegression(max_iter=2000), parameters)
	grid_search.fit(train_features, train_labels)

	log.info(f'Best parameters: \n { grid_search.best_params_}')
	log.info(f'Best scrores: \n {grid_search.best_score_}')

	# train using best params
	lr_clf = LogisticRegression(
		C=grid_search.best_params_['C'],
		fit_intercept=grid_search.best_params_['fit_intercept'],
		solver=grid_search.best_params_['solver'],
		max_iter=2000
	)
else:

	optimal_params= {
		'C': 5.263252631578947,
		'fit_intercept': True,
		'solver': 'saga',
		'verbose': 0
	}
	base_params = {
		'C': 12.50075,
		'fit_intercept': True,
		'solver': 'saga',
		'verbose': 0
	}

	# train using base params
	lr_clf = LogisticRegression(
		C=optimal_params['C'],
		fit_intercept=optimal_params['fit_intercept'],
		solver=optimal_params['solver'],
		max_iter=2000
	)

# fit model
lr_clf.fit(train_features, train_labels)

# label test data
y_pred_binary = lr_clf.predict(test_features)
y_pred_prob = lr_clf.predict_proba(test_features)

# compute metrics
score = lr_clf.score(test_features, test_labels)
cfm = metrics.confusion_matrix(test_labels, y_pred_binary)
precision = metrics.precision_score(test_labels, y_pred_binary)
log_loss = metrics.log_loss(test_labels.astype('float'), y_pred_prob)

# save model
pickle.dump(
	lr_clf,
	open(f'''models/logistic/older/{datetime.now().strftime('%d-%b-%y_%H:%M:%S')}_log_loss-{log_loss}_score-{score}.pkl''', 'wb')
)
pickle.dump(lr_clf, open(f'''models/logistic/latest/latest.pkl''', 'wb'))

print(f'Accuracy: {score}')
print(f'Confusion Matrix: {cfm}')
print(f'Precision: {precision}')
print(f'Log Loss: {log_loss}')

# classify test data
main_test_labelled = lr_clf.predict_proba(main_test_features)
main_test_labelled_df = pd.DataFrame(main_test_labelled)
main_test_data['target'] = main_test_labelled_df[[1]]
main_test_predictions = main_test_data[['ID', 'target']]
main_test_predictions.to_csv(
	f'''data/output/older/{datetime.now().strftime('%d-%b-%y_%H:%M:%S')}_log_loss-{log_loss}_score-{score}.csv''',
	index=False
)
main_test_predictions.to_csv(f'''data/output/latest/latest.csv''', index=False)
