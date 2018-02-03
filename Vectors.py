##created by Runqian Huang, 11/5/2017
# from gensim.models import word2vec

#create array of vectors
#call cosine similarity to find word for question
import gensim
import numpy as np


class classifier(object):
    def __init__(self):

        self.model = gensim.models.KeyedVectors.load_word2vec_format('model.bin')
        self.sentence_vectors = []
        self.question_vectors = []
        # self.sentence_vector = self.create_avg_vector(sentence, self.sentence_vectors)
        # self.question_vector = self.create_avg_vector(question, self.question_vectors)

    def create_avg_vector(self, text):
        w = np.zeros(300)
        word_vectors = self.model.wv
        numZeroVec = 0
        count = 0

        # create word level vector
        v = np.zeros(300)
        for word in text:
            if word in word_vectors.vocab:
                count += 1
                v += word_vectors[word]
        if count != 0:
            v /= count
            w += v
        else:
            numZeroVec += 1

        w /= (len(text) - numZeroVec)
        # print w

        return w

    def cosine_similarity(self, v, u):
        return np.inner(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))

    def result(self):
        score = self.cosine_similarity(self.sentence_vector, self.question_vector)
        return score
