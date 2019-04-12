import pickle as cPickle

if __name__ == '__main__':
    with open("../data/embeddings/ent2embed.pk", "rb") as rf:
        ent2embed = cPickle.load(rf)