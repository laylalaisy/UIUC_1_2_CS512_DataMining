import models
from dataset import Dataset


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

    # Prior Model
    model = models.PriorModel()
    trainset = Dataset.get('train')
    model.fit(trainset)
    print('Training finished!')

    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))