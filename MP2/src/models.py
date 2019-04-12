import random
import numpy as np
import random
import textdistance
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

class RandomModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        # fill this function if your model requires training
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(random.choice(mention.candidates).id if mention.candidates else 'NIL')
        return pred_cids


''' 
*** Model Name: PriorModel ***
*** Model Function: returns the candidate with the highest prior probability as the prediction for each mention ***
'''
class PriorModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(mention.candidates[0].id if mention.candidates else 'NIL')
        return pred_cids


''' 
*** Model Name: SupModel ***
*** Model Function: use a simple supervised model ***
'''
class SupModel:
    def __init__(self):
        self.model = None

    def fit(self, dataset):
        X = []
        y = []  # label

        for mention in dataset.mentions:
            surface = mention.surface

            if mention.candidates:
                # add feature values
                i = 0
                candidates_name = []
                x = np.zeros(24)  # feature
                for candidate in mention.candidates:
                    # feature1: prob
                    x[i] = candidate.prob
                    i += 1
                    # feature2: similarity
                    x[i] = textdistance.hamming.normalized_similarity(surface, candidate.name)
                    i += 1
                    # feature2: similarity
                    x[i] = textdistance.hamming.normalized_distance(surface, candidate.name)
                    i += 1
                    candidates_name.append(candidate.name)
                X.append(x)

                try:
                    y.append(candidates_name.index(mention.gt.name))
                except ValueError:
                    y.append(-1)
            else:
                y.append(-1)

        X = np.array(X)
        y = np.array(y)
        # self.model = SVC().fit(X, y)
        # self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, y)
        self.model = RandomForestClassifier().fit(X, y)



    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            surface = mention.surface

            if mention.candidates:
                # add feature values
                i = 0
                x = np.zeros(24)  # feature
                for candidate in mention.candidates:
                    # feature1: prob
                    x[i] = candidate.prob
                    i += 1
                    # feature2: similarity
                    x[i] = textdistance.hamming.normalized_similarity(surface, candidate.name)
                    i += 1
                    # feature2: similarity
                    x[i] = textdistance.hamming.normalized_distance(surface, candidate.name)
                    i += 1
                predict_idx = self.model.predict(x.reshape(1, -1))[0]
                if predict_idx == -1:
                    pred_cids.append(mention.candidates[0].id)
                else:
                    pred_cids.append(mention.candidates[predict_idx].id)
            else:
                pred_cids.append('NIL')

        return pred_cids









