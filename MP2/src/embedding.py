import pickle as cPickle

if __name__ == '__main__':
    with open("../data/embeddings/ent2embed.pk", "rb") as rf1:
        ent2embed = cPickle.load(rf1)
    with open("../data/embeddings/word2embed.pk", "rb") as rf2:
        word2embed = cPickle.load(rf2)

    word = 'U.S.'

    try:
        print(ent2embed[word])
    except KeyError:
        print("ent2embed does not work, bro!")

    try:
        print(word2embed[word])
    except KeyError:
        print("word2embed does not work, bro!")
