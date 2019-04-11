from pathlib import Path

DIR_PROJ = Path(__file__).absolute().parent.parent
DIR_DATA = DIR_PROJ / 'data'
DIR_DATASET = DIR_DATA / 'test_train'
DIR_EMBED = DIR_DATA / 'embeddings'
PATH_E2V = DIR_EMBED / 'ent2embed.pk'
PATH_W2V = DIR_EMBED / 'word2embed.pk'