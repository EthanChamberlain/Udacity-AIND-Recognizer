import numpy as np
import pandas as pd
from asl_data import AslDb
from asl_utils import test_features_tryit
from asl_utils import test_std_tryit
#part 2
import warnings
from hmmlearn.hmm import GaussianHMM
import math
from matplotlib import (cm, pyplot as plt, mlab)
from my_model_selectors import SelectorCV
import timeit
# autoreload for automatically reloading changes made in my_model_selectors and my_recognizer
#%load_ext autoreload
#%autoreload 2
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
asl = AslDb() # initializes the database

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']



# training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
# sequences = training.get_all_sequences()
# Xlengths = training.get_all_Xlengths()
# for word in words_to_train:
#     start = timeit.default_timer()
#     model = SelectorCV(sequences, Xlengths, word, 
#                     min_n_components=2, max_n_components=15, random_state = 14).select()
#     end = timeit.default_timer()-start
#     if model is not None:
#         print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
#     else:
#         print("Training failed for {}".format(word))

# from my_model_selectors import SelectorBIC

# training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
# sequences = training.get_all_sequences()
# Xlengths = training.get_all_Xlengths()
# for word in words_to_train:
#     start = timeit.default_timer()
#     model = SelectorBIC(sequences, Xlengths, word, 
#                     min_n_components=2, max_n_components=15, random_state = 14).select()
#     end = timeit.default_timer()-start
#     if model is not None:
#         print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
#     else:
#         print("Training failed for {}".format(word))

# from my_model_selectors import SelectorDIC

# training = asl.build_training(features_ground) 
# sequences = training.get_all_sequences()
# Xlengths = training.get_all_Xlengths()
# #print(Xlengths.keys())
# for word in words_to_train:
#     start = timeit.default_timer()
#     model = SelectorDIC(sequences, Xlengths, word, 
#                     min_n_components=2, max_n_components=15, random_state = 14).select()
#     end = timeit.default_timer()-start
#     if model is not None:
#         print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
#     else:
#         print("Training failed for {}".format(word))
import unittest
from asl_test_model_selectors import TestSelectors
suite = unittest.TestLoader().loadTestsFromModule(TestSelectors())
unittest.TextTestRunner().run(suite)
