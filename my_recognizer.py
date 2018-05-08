import warnings
from asl_data import SinglesData
'part3'
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
from my_model_selectors import SelectorConstant
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorCV
from asl_utils import show_errors
asl = AslDb() # initializes the database



#asl.df.ix[98,1]  # look at the data available for an individual frame

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']

# collect the features into a list
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

#finding means of all speaker subgroups
df_means = asl.df.groupby('speaker').mean()
#means that match by speaker
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])

#asl.df.head()
#standard deviations grouped by speaker
df_std = asl.df.groupby('speaker').std()
#stds that match by speaker
asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])


# Z score = # of standard deviations from mean
# Name these 'norm-rx', 'norm-ry', 'norm-lx', and 'norm-ly'
# using Z-score scaling (X-Xmean)/Xstd
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

# TODO add features for polar coordinate values where the nose is the origin
# Name these 'polar-rr', 'polar-rtheta', 'polar-lr', and 'polar-ltheta'
# Note that 'polar-rr' and 'polar-rtheta' refer to the radius and angle
#r = sqrt(x^2 + y^2)

asl.df['polar-rr'] = np.sqrt(pow(asl.df['grnd-ry'],2) + pow(asl.df['grnd-rx'],2))
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'],asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt(pow(asl.df['grnd-ly'],2) + pow(asl.df['grnd-lx'],2))
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'],asl.df['grnd-ly'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'
#use pandas diff and fillna methods
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-rx'] =  asl.df['right-x'].diff().fillna(0)
asl.df['delta-lx'] =  asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] =  asl.df['left-y'].diff().fillna(0)
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

#Custom features chosen were differences between hands, and differences between hands over time
#which i thought would rule out many signs(eg deciding between a sign where the hands approach one another or run away from one another)
#perhaps a better custom measurement would be to do this incorporating the ground values so that the nose is factored in as well.
#difference between hands
asl.df['hands-x'] = asl.df['right-x'] - asl.df['left-x']
asl.df['hands-y'] = asl.df['right-y'] - asl.df['left-y']
asl.df['delta-hands-x'] =  asl.df['hands-x'].diff().fillna(0)
asl.df['delta-hands-y'] =  asl.df['hands-y'].diff().fillna(0)

# TODO define a list named 'features_custom' for building the training set
features_custom = ['hands-x','hands-y','delta-hands-x','delta-hands-y']


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for k in test_set.get_all_Xlengths():
        scores = {}
        X,lengths = test_set.get_item_Xlengths(k)
        for word, model in models.items():
            try:
                logL = model.score(X,lengths)
                scores[word] = logL
            except:
                scores[word] = float("-inf")
                
        probabilities.append(scores)
        guesses.append(max(scores,key = scores.get))
    return probabilities, guesses


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

 

# all_features =[features_ground,features_polar,features_delta,features_custom]
# selectors = [SelectorBIC,SelectorDIC,SelectorCV]
# import sys

# sys.stdout = open('results.txt','w')
# i = 0
# for f in all_features:
#     for s in selectors:
#         i += 1
#         print('\n')
#         print('Combo: ' + str(i) + ' ')
#         print(str(f))
#         print(str(s))
#         print('\n')
#         models = train_all_words(f,s)
#         test_set = asl.build_test(f)
#         probabilities, guesses = recognize(models,test_set)
#         print('probabilities: ')
#         print(str(probabilities))
#         print(' guesses: ')
#         print(str(guesses))
#         show_errors(guesses,test_set)

