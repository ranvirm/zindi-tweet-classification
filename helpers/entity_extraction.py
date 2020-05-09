import spacy as sp
from pandas import DataFrame
from sklearn_pandas import DataFrameMapper


def extract_entities(data_frame, text_col):
	assert type(data_frame) == DataFrame, 'Input data_frame is not of type DataFrame'

	nlp = sp.load("en_core_web_sm")

	def extract_entities_tuple_list(s, nlp_model):
		s_classified = nlp_model(s)
		return [(X.label_, X.text) for X in s_classified.ents]

	def extract_entities_dict_list(s, nlp_model):
		y = nlp_model(s)
		entities_dict = {
			'PERSON': [],
			'NORP': [],
			'FAC': [],
			'ORG': [],
			'GPE': [],
			'LOC': [],
			'PRODUCT': [],
			'EVENT': [],
			'WORK_OF_ART': [],
			'LAW': [],
			'LANGUAGE': [],
			'DATE': [],
			'TIME': [],
			'PERCENT': [],
			'MONEY': [],
			'QUANTITY': [],
			'ORDINAL': [],
			'CARDINAL': [],
		}
		for i in y.ents:
			for ii in entities_dict.keys():
				if i.label_ == ii:
					entities_dict[ii].append(i.text)
		return entities_dict

	'''
	PERSON	People, including fictional.
	NORP	Nationalities or religious or political groups.
	FAC	Buildings, airports, highways, bridges, etc.
	ORG	Companies, agencies, institutions, etc.
	GPE	Countries, cities, states.
	LOC	Non-GPE locations, mountain ranges, bodies of water.
	PRODUCT	Objects, vehicles, foods, etc. (Not services.)
	EVENT	Named hurricanes, battles, wars, sports events, etc.
	WORK_OF_ART	Titles of books, songs, etc.
	LAW	Named documents made into laws.
	LANGUAGE	Any named language.
	DATE	Absolute or relative dates or periods.
	TIME	Times smaller than a day.
	PERCENT	Percentage, including ”%“.
	MONEY	Monetary values, including unit.
	QUANTITY	Measurements, as of weight or distance.
	ORDINAL	“first”, “second”, etc.
	CARDINAL	Numerals that do not fall under another type.
	'''

	entities = [
		'PERSON',
		'NORP',
		'FAC',
		'ORG',
		'GPE',
		'LOC',
		'PRODUCT',
		'EVENT',
		'WORK_OF_ART',
		'LAW',
		'LANGUAGE',
		'DATE',
		'TIME',
		'PERCENT',
		'MONEY',
		'QUANTITY',
		'ORDINAL',
		'CARDINAL',
	]

	data_frame['named_entities_tuple_list'] = data_frame[text_col].apply(lambda x: extract_entities_tuple_list(x, nlp))
	data_frame['named_entities_dict_list'] = data_frame[text_col].apply(
		lambda x: extract_entities_dict_list(x, nlp))

	for e in entities:
		data_frame[e] = data_frame.named_entities_dict_list.apply(lambda x: x[e])

	bin_cols = []
	# add binary cols
	for e in entities:
		data_frame[f'contains_{e.lower()}'] = data_frame.named_entities_dict_list.apply(lambda x: 1 if len(x[e]) > 0 else 0)
		bin_cols.append(f'contains_{e.lower()}')

	mapper = DataFrameMapper([
		(bin_cols, None)
	])
	x = mapper.fit_transform(data_frame)

	return x
