from helpers import entity_extraction
import pandas as pd
import numpy as np

data = pd.read_csv('data/sources/train-2.csv')[:20]

df = entity_extraction.extract_entities(data, 'text')

assert df == pd.DataFrame

cols = ['contains_person', 'contains_norp',
       'contains_fac', 'contains_org', 'contains_gpe', 'contains_loc',
       'contains_product', 'contains_event', 'contains_work_of_art',
       'contains_law', 'contains_language', 'contains_date', 'contains_time',
       'contains_percent', 'contains_money', 'contains_quantity',
       'contains_ordinal', 'contains_cardinal']

df1 = df[cols].apply(lambda x: [np.asarray(i) for i in np.asarray(x)], axis=1)

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
CARDINAL	Numerals that do not fall under another type.'''


