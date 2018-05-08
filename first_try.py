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



def train_a_word(word, num_hidden_states, features):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)  
    X, lengths = training.get_word_Xlengths(word)
    for i in X:
        print("X: " + i + " " +  X[i])
    for l in lengths:
        print("lengths: " + l + " " + lengths[i])
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    
    return model, logL

def show_model_stats(word, model):
    print("Number of states trained in model for {} is {}".format(word, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):  # for each hidden state
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        print()
    
def visualize(word, model):
    """ visualize the input model for a particular word """
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])
    figures = []
    for parm_idx in range(len(model.means_[0])):
        xmin = int(min(model.means_[:,parm_idx]) - max(variance[:,parm_idx]))
        xmax = int(max(model.means_[:,parm_idx]) + max(variance[:,parm_idx]))
        fig, axs = plt.subplots(model.n_components, sharex=True, sharey=False)
        colours = cm.rainbow(np.linspace(0, 1, model.n_components))
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            x = np.linspace(xmin, xmax, 100)
            mu = model.means_[i,parm_idx]
            sigma = math.sqrt(np.diag(model.covars_[i])[parm_idx])
            ax.plot(x, mlab.normpdf(x, mu, sigma), c=colour)
            ax.set_title("{} feature {} hidden state #{}".format(word, parm_idx, i))

            ax.grid(True)
        figures.append(plt)
    for p in figures:
        p.show()
        

demoword = 'BOOK'
model, logL = train_a_word(demoword, 3, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))

model, logL = train_a_word(demoword, 4, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))

model, logL = train_a_word(demoword, 1, features_ground)
print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
print("logL = {}".format(logL))


# show_model_stats(demoword, model)
# my_testword = 'CHOCOLATE'
# model, logL = train_a_word(my_testword, 3, features_ground) # Experiment here with different parameters
# show_model_stats(my_testword, model)
# print("logL = {}".format(logL))

# print("Number of states trained in model for {} is {}".format(word, model.n_components))

# #visualize(my_testword, model)

# training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
# word = 'VEGETABLE' # Experiment here with different words
# model = SelectorConstant(training.get_all_sequences(), training.get_all_Xlengths(), word, n_constant=3).select()

# training = asl.build_training(features_ground)
# print("Training words: {}".format(training.words))

# #DATA CAN BE ACCESSED WITH
# #get_all_sequences, get_all_Xlengths, get_word_sequences,
# #and get_word_Xlengths
# training.get_word_Xlengths('CHOCOLATE')

# #show a single set of features for a given (video, frame) tuple
# [asl.df.ix[98,1][v] for v in features_ground]


#asl.df.head() # displays the first five rows of the asl database, indexed by video and frame



# test the code
#test_std_tryit(df_std)
#test_features_tryit(asl)

# import unittest
# # import numpy as np

# class TestFeatures(unittest.TestCase):

#     def test_features_ground(self):
#         sample = (asl.df.ix[98, 1][features_ground]).tolist()
#         self.assertEqual(sample, [9, 113, -12, 119])

#     def test_features_norm(self):
#         sample = (asl.df.ix[98, 1][features_norm]).tolist()
#         np.testing.assert_almost_equal(sample, [ 1.153,  1.663, -0.891,  0.742], 3)

#     #test failed    
#     def test_features_polar(self):
#         sample = (asl.df.ix[98,1][features_polar]).tolist()
#         print(sample)
#         np.testing.assert_almost_equal(sample, [113.3578, 0.0794, 119.603, -0.1005], 3)

#     def test_features_delta(self):
#         sample = (asl.df.ix[98, 0][features_delta]).tolist()
#         self.assertEqual(sample, [0, 0, 0, 0])
#         sample = (asl.df.ix[98, 18][features_delta]).tolist()
#         self.assertTrue(sample in [[-16, -5, -2, 4], [-14, -9, 0, 0]], "Sample value found was {}".format(sample))
                         
# suite = unittest.TestLoader().loadTestsFromModule(TestFeatures())
# unittest.TextTestRunner().run(suite)
