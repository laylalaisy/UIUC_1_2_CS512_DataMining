import random
import numpy as np
import random
import textdistance

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
        pass

    def fit(self, dataset):
        for mention in dataset.mentions:
            if mention.surface:
                surface = mention.surface

                x = np.zeros(16)    # feature
                y = []              # label

                i = 0
                if mention.candidates:

                    # add feature values
                    candidates_name = []
                    for candidate in mention.candidates:
                        if len(candidate.name)>0:
                            # feature1: similarity
                            x[i] = textdistance.hamming.similarity(surface, candidate.name)
                            i += 1
                            # feature2: prob
                            x[i] = candidate.prob
                            i += 1
                            candidates_name.append(candidate.name)

                    # add label
                    try:
                        label_index = candidates_name.index(mention.gt.name)

                    except ValueError:
                        label_index = y.append(random.randint(0, 8))
                    y.append(label_index)
                

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(mention.gt.id if mention.gt else 'NIL')
        return pred_cids



