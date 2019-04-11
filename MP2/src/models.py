import random
import models
from dataset import Dataset


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


if __name__ == '__main__':
    # Prior Model
    model = models.PriorModel()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('Training finished!')

    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))



