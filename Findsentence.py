##created by Runqian Huang, 11/5/2017

##R keeps vectorizing the whole corpus for every question
import json
import time
import string
import re
import Vectors as V

##normalize the strings to improve the results
def normalize(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

##transform dictionary to json file
def store(data):
    with open('findsentence.json', 'w') as json_file:
        json_file.write(json.dumps(data))

##transform json file to dictionary
def load():
    with open('training.json') as json_file:
        data = json.load(json_file)
        return data

if __name__ == "__main__":

    data = {}
    data = load()
    dataset = data['data']
    result = {}
    classifier = V.classifier()
    ##here we compute unigram overlap, because the order of the words in the questions are often different from contexts, unigram is more reliable
    for article in dataset:
        for paragraph in article['paragraphs']:
            paragraph_tokens=paragraph['context'].split() ##get tokens from context
            numofsen=0  ##number of sentences
            sentence={}
            sentence[numofsen] = ([])
            for x in paragraph_tokens:
                sentence[numofsen].append(normalize(x))
                if '.'in x or ',' in x or '?' in x or '!' in x or ';' in x or ':' in x:  ##sign as the end of the sentence
                    numofsen+=1
                    sentence[numofsen] = ([])

            #print('start_classifier')

            #print('classifier started')

            #iterate through array sentences
            #for each sentence, create average vector
            #put average vector that's returned into a new array (of vectors)

            sentence_vectors = []

            for i in range(0,numofsen):
                sentence_vectors.append(classifier.create_avg_vector(sentence[i]))
                print('sentence_to_vector')

            for qa in paragraph['qas']: ##for each question, compute the vector
                maxsim = -1 ##maximum of the similarity
                maxnum = -1 ##the sentence with the maximal similarity
                startoverlap = -1  ##start position of the overlap
                endoverlap = -1  ##end position of the overlap
                qa_tokens=normalize(qa['question']).split()
                qa_vector = classifier.create_avg_vector(qa_tokens)
                for i in range(0,numofsen):
                    total=0
                    start=-1
                    end=-1
                    total= classifier.cosine_similarity(sentence_vectors[i], qa_vector)##compute the similarity
                    if total>=maxsim: ##find the maximum similarity
                        maxsim=total
                        maxnum=i
                        for j, x in enumerate(sentence[i]): ##find the start and the end position of the overlap
                            if x in qa_tokens:
                                if start == -1:
                                    start = j
                                end = j
                        startoverlap=start
                        endoverlap=end
                result[qa['id']]=([])
                result_str=''
                if(maxnum==-1):
                    result_str=''
                else:
                    for i,x in enumerate(sentence[maxnum]): ##output the sentence with maximal similarity
                        if i<startoverlap:
                            result_str += x + ' '
                        elif i>endoverlap:
                            result_str += x + ' '
                result[qa['id']]=result_str
    store(result) ##write the answers to json file