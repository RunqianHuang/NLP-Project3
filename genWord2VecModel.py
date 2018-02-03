##created by Runqian Huang, 11/5/2017
from gensim.models import word2vec
import logging
import os.path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size=300)
model.wv.save_word2vec_format("model.bin")

