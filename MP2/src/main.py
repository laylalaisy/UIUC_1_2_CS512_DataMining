import models
from dataset import Dataset
import pickle as cPickle

if __name__ == '__main__':
    # # Random Model
    # model = models.RandomModel()
    # trainset = Dataset.get('train')
    # model.fit(trainset)
    # print('Training finished!')
    #
    # for dsname in Dataset.ds2path.keys():
    #     ds = Dataset.get(dsname)
    #     pred_cids = model.predict(ds)
    #     print(dsname, ds.eval(pred_cids))

    # # Prior Model
    # model = models.PriorModel()
    # trainset = Dataset.get('train')
    # model.fit(trainset)
    # print('Training finished!')
    #
    # for dsname in Dataset.ds2path.keys():
    #     ds = Dataset.get(dsname)
    #     pred_cids = model.predict(ds)
    #     print(dsname, ds.eval(pred_cids))

    with open("../data/embeddings/ent2embed.pk", "rb") as rf:
        ent2embed = cPickle.load(rf)

    # SupModel
    model = models.SupModel()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('Training finished!')

    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))

