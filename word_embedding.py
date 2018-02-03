#from gensim.models import word2vec
import gensim
import numpy as np

class classifier(object):

	def __init__(self,p_train_text,n_train_text):
		
		self.model = gensim.models.KeyedVectors.load_word2vec_format('model.bin')
		self.p_vectors = []
		self.n_vectors = []
		self.p_vector = self.create_avg_vector(p_train_text,self.p_vectors)
		self.n_vector = self.create_avg_vector(n_train_text,self.n_vectors)

	def create_avg_vector(self,text,s):
		w = np.zeros(500)
		word_vectors = self.model.wv
		numZeroVec = 0
		for line in text:
			line = line.lower().split(' ')
			count = 0

			#create word level vector
			v = np.zeros(500)
			for word in line:
				if word in word_vectors.vocab:
					count += 1
					v += word_vectors[word]
			if count != 0:
				v /= count
				s.append(v)
				w += v
			else: 
				print("this one was excluded " + str(line))
				numZeroVec += 1
		
		w /= (len(text) - numZeroVec)
		#print w

		return w

	def cosine_similarity(self,v,u):
		return np.inner(v,u)/(np.linalg.norm(v)*np.linalg.norm(u))

	def vectorize(self,line): 
		line = line.lower().split(' ')
		word_vectors = self.model.wv
		v = np.zeros(500)
		count = 0

		for word in line:
			if word in word_vectors.vocab:
				count += 1
				v += word_vectors[word]

		if count != 0:
			return v/count
		else:
			return np.zeros(500)

	def classify(self,v):
		p_score = self.cosine_similarity(v,self.p_vector)
		n_score = self.cosine_similarity(v,self.n_vector)

		if p_score < n_score:
			return 1
		else:
			return 0

	def betterClassify(self, u):
		closest = self.cosine_similarity(u,self.p_vectors[:1])
		closestClass = 0

		for v in self.p_vectors[1:]:
			similarity = self.cosine_similarity(u,v)
			if similarity < closest:
				closest = similarity

		for v in self.n_vectors:
			similarity = self.cosine_similarity(u,v)
			if similarity < closest:
				closest = similarity
				closestClass = 1

		return closestClass


