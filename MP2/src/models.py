import random
import numpy as np
import random
import scipy
import textdistance
import pandas as pd
from itertools import chain
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

''' 
*** Model Name: EmbeddingModel ***
*** Model Function: make use of the entity embeddings. For each candidate entity:***
        - concatenate the hand-crafted features created in step2 
        - the entity embedding of the candidate (300 dimensions)
        - the sum of the context word embeddings of the mention (300 dimensions)
        - the cosine similarity between the two embeddings (1 dimension)
'''
class EmbeddingModel:
    def __init__(self, ent2embed, word2embed):
        self.ent2embed = ent2embed
        self.word2embed = word2embed
        self.model = None

    def fit(self, dataset):
        X = []
        y = []
        for mention in dataset.mentions:
            surface = mention.surface

            # surface embedding
            try:
                if " " in surface:
                    temp_surface = surface.replace(" ", "_")
                    surface_embedding = self.ent2embed[temp_surface]
                else:
                    surface_embedding = self.word2embed[surface]
            except:
                continue

            # sum of contexts embedding
            sum_contexts = np.zeros(300)
            for word in mention.contexts:
                try:
                    sum_contexts += np.array(self.ent2embed[word])
                except:
                    try:
                        sum_contexts += np.array(self.word2embed[word])
                    except:
                        continue
            sum_contexts = list(sum_contexts)


            if mention.candidates:
                for candidate in mention.candidates:
                    # feature
                    x = []
                    temp_candidate = candidate.name.replace(" ", "_")
                    candidate_embedding = self.ent2embed[temp_candidate]
                    candidate_embedding_list = list(candidate_embedding)
                    # feature1: prob
                    x.append(candidate.prob)
                    # feature2: similarity
                    x.append(textdistance.hamming.normalized_similarity(surface, candidate.name))
                    # feature3: similarity
                    x.append(textdistance.hamming.normalized_distance(surface, candidate.name))
                    # feature4: the entity embedding of the candidate (300 dimensions)
                    for i in candidate_embedding_list:
                        x.append(i)
                    # feature5: the sum of the context word embeddings of the mention (300 dimensions)
                    for i in sum_contexts:
                        x.append(i)
                    # feature6: the cosine similarity between the two embeddings (1 dimension)
                    x.append(scipy.spatial.distance.cosine(surface_embedding, candidate_embedding))

                    x_pd = pd.DataFrame(x)
                    if x_pd.isnull().values.any():
                        continue
                    else:
                        X.append(x)
                        # label
                        if candidate.name == mention.gt.name:
                            y.append(1)
                        else:
                            y.append(0)

        X = np.array(X)
        self.model = RandomForestClassifier().fit(X, y)



    def predict(self, dataset):
        pred_cids = []

        X = []
        for mention in dataset.mentions:
            surface = mention.surface

            # surface embedding
            try:
                if " " in surface:
                    temp_surface = surface.replace(" ", "_")
                    surface_embedding = self.ent2embed[temp_surface]
                else:
                    surface_embedding = self.word2embed[surface]
            except:
                try:
                    pred_cids.append(mention.candidates[0].id)
                except:
                    pred_cids.append('NIL')
                continue


            # sum of contexts embedding
            sum_contexts = np.zeros(300)
            for word in mention.contexts:
                try:
                    sum_contexts += np.array(self.ent2embed[word])
                except:
                    try:
                        sum_contexts += np.array(self.word2embed[word])
                    except:
                        continue
            sum_contexts = list(sum_contexts)

            if mention.candidates:
                is_exist = False
                for candidate in mention.candidates:
                    # feature
                    x = []
                    temp_candidate = candidate.name.replace(" ", "_")
                    candidate_embedding = self.ent2embed[temp_candidate]
                    candidate_embedding_list = list(candidate_embedding)
                    # feature1: prob
                    x.append(candidate.prob)
                    # feature2: similarity
                    x.append(textdistance.hamming.normalized_similarity(surface, candidate.name))
                    # feature3: similarity
                    x.append(textdistance.hamming.normalized_distance(surface, candidate.name))
                    # feature4: the entity embedding of the candidate (300 dimensions)
                    for i in candidate_embedding_list:
                        x.append(i)
                    # feature5: the sum of the context word embeddings of the mention (300 dimensions)
                    for i in sum_contexts:
                        x.append(i)
                    # feature6: the cosine similarity between the two embeddings (1 dimension)
                    x.append(scipy.spatial.distance.cosine(surface_embedding, candidate_embedding))

                    x_pd = pd.DataFrame(x)
                    if x_pd.isnull().values.any():
                        continue
                    else:
                        # label
                        x = np.array(x)
                        prediction = self.model.predict(x.reshape(1, -1))[0]
                        if prediction == 1:
                            pred_cids.append(candidate.id)
                            is_exist = True
                            break
                        else:
                            continue

                # all prediction is 0
                if is_exist == False:
                    pred_cids.append(mention.candidates[0].id)
            else:
                pred_cids.append('NIL')

        return pred_cids









