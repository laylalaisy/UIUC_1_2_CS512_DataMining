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
    #
    # # SupModel
    # model = models.SupModel()
    # trainset = Dataset.get('train')
    # model.fit(trainset)
    # print('Training finished!')
    #
    # for dsname in Dataset.ds2path.keys():
    #     ds = Dataset.get(dsname)
    #     pred_cids = model.predict(ds)
    #     print(dsname, ds.eval(pred_cids))


    with open("../data/embeddings/ent2embed.pk", "rb") as rf1:
        ent2embed = cPickle.load(rf1)
    with open("../data/embeddings/word2embed.pk", "rb") as rf2:
        word2embed = cPickle.load(rf2)

    # Embedding Model
    model = models.EmbeddingModel(ent2embed, word2embed)
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('Training finished!')

    # for dsname in Dataset.ds2path.keys():
    #     ds = Dataset.get(dsname)
    #     pred_cids = model.predict(ds)
    #     print(dsname, ds.eval(pred_cids))


