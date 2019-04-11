from util import Json
from const import DIR_DATASET


class Candidate:
    def __init__(self, cid, prob, name):
        self.id = cid
        self.prob = prob
        self.name = name


class Mention:
    def __init__(self, m_dict):
        self.fileid = m_dict['fid']
        self.surface = m_dict['surface']
        self.contexts = [c.split() for c in m_dict['contexts']]
        self.candidates = [Candidate(i, p, n) for i, p, n in m_dict['candidates']]
        g_id, g_prob, g_name = m_dict['gt']
        self.gt = Candidate(g_id, g_prob, g_name)


class Dataset:
    trainpath = DIR_DATASET / 'aida_train.json'
    ds2path = {
        'aidaA': DIR_DATASET / 'aida_testA.json',
        'aidaB': DIR_DATASET / 'aida_testB.json',
        'msnbc': DIR_DATASET / 'wned-msnbc.json',
        'ace': DIR_DATASET / 'wned-ace2004.json',
        'aquaint': DIR_DATASET / 'wned-aquaint.json',
    }

    def __init__(self, filepath):
        self.mentions = [Mention(d) for d in Json.loadf(filepath)]

    @classmethod
    def get(cls, dsname):
        path = cls.trainpath if dsname == 'train' else cls.ds2path[dsname]
        return Dataset(path)

    def eval(self, pred_cids):
        gold_cids = [m.gt.id for m in self.mentions]
        
        true_pos = 0
        for g, p in zip(gold_cids, pred_cids):
            true_pos += int(g == p and p != 'NIL')

        precision = true_pos / len([p for p in pred_cids if p != 'NIL'])
        recall = true_pos / len(gold_cids)
        f1 = 2 * precision * recall / (precision + recall)
        return f1
