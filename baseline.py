##created by Runqian Huang, 10/27/2017
import json
import time
import string
import re

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
    with open('baselineresult.json', 'w') as json_file:
        json_file.write(json.dumps(data))

##transform json file to dictionary
def load():
    with open('development.json') as json_file:
        data = json.load(json_file)
        return data

if __name__ == "__main__":

    data = {}
    data = load()
    dataset = data['data']
    result = {}
    ##here we compute unigram overlap, because the order of the words in the questions are often different from contexts, unigram is more reliable
    for article in dataset:
        for paragraph in article['paragraphs']:
            paragraph_tokens=paragraph['context'].split() ##get tokens from context
            numofsen=0  ##number of sentences
            sentence={}
            sentence[numofsen] = ([])
            for x in paragraph_tokens:
                sentence[numofsen].append(normalize(x))
                if '.'in x or ','in x or '?'in x or '!'in x or ';'in x or ':'in x:  ##if reach any punctuation, sign as the end of the sentence, or the sentence will be very long
                    numofsen+=1
                    sentence[numofsen] = ([])
            for qa in paragraph['qas']: ##for each question, compute the unigram overlap
                maxoverlap = -1 ##maximum of the number of overlap tokens
                maxnum = -1 ##the sentence with the most overlap
                startoverlap = -1 ##start position of the overlap
                endoverlap = -1 ##end position of the overlap
                qa_tokens=normalize(qa['question']).split()
                for i in range(0,numofsen):
                    total=0
                    start=-1
                    end=-1
                    for x in qa_tokens: ##count the overlap tokens
                        if x in sentence[i]:
                            total+=1
                    if total>=maxoverlap: ##find the maximum overlap
                        maxoverlap=total
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
                    for i,x in enumerate(sentence[maxnum]): ##use the part before the start of the overlap and the part after the end of the overlap as the answer to avoid too long answers
                         if i<startoverlap:                    #and the answers are approximately arround the overlap
                             result_str += x + ' '
                         if i>endoverlap:
                             result_str += x + ' '
                result[qa['id']]=result_str
    store(result) ##write the answers to json file

    #####################################################################################################
    ##             The result of development set: "exact_match": 5.695,  "f1": 21.950                  ##
    ##             The result of training set: "exact_match": 4.351,  "f1": 19.444                     ##
    ##             The result of testing set: "exact_match": 4.037,  "f1": 17.984                  ##
    #####################################################################################################