from .training_data import TrainingData
from .training import Trainer
from gensim.models import Word2Vec


class SkipAtomModel:
    def __init__(self, training_data, embeddings):
        self.vectors = embeddings
        self.dictionary = training_data.atom_to_index

    @staticmethod
    def load(model_file, training_data_file):
        td = TrainingData.load(training_data_file)
        embeddings = Trainer.load_embeddings(model_file)
        return SkipAtomModel(td, embeddings)


class Word2VecModel:
    def __init__(self, w2v_model):
        wv =  w2v_model.wv
        self.vectors = wv.vectors
        self.dictionary = {w: wv.vocab[w].index for w in wv.vocab}

    @staticmethod
    def load(model_file):
        return Word2VecModel(Word2Vec.load(model_file))
