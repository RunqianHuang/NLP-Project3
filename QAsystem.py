import json
import nltk
from nltk.chunk import conlltags2tree, tree2conlltags
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

    return white_space_fix(remove_articles(remove_punc(s)))

##transform dictionary to json file
def store(data):
    with open('result.json', 'w') as json_file:
        json_file.write(json.dumps(data))

##transform json file to dictionary
def load():
    with open('testing.json') as json_file:
        data = json.load(json_file)
        return data

##Uses NER tags and question categorizaion
def better_qa(tag, context, numsen, qid, question, entities, pos, result):
    #print tag
    if tag == "OTHER1":
        #print "tag is OTHER"
        sliding_window1(context, numsen, qid, question, pos, result)
    elif tag == "OTHER2":
        #print "tag is OTHER"
        sliding_window2(context, numsen, qid, question, pos, result)
    else:
        result[qid] = ([])
        result_str = ""
        for word in context:
            #print('in better qa: '+ word)
            word = word.encode("utf-8")
            added = False
            if tag == "LOC":
                if "LOCATION" in entities:
                    if word in entities["LOCATION"]:
                        #print "Word has tag LOCATION"
                        result_str += word + ' '
                        added = True
                elif "GPE" in entities:
                    if word in entities["GPE"] and not added:
                        #print "Word has tag GPE"
                        result_str += word + ' '
                        added = True
            elif tag == "TIM":
                if "TIME" in entities:
                    if word in entities["TIME"]:
                        #print "Word has tag TIME"
                        result_str += word + ' '
                        added = True
                elif "DATE" in entities:
                    if word in entities["DATE"] and not added:
                        #print "Word has tag DATE"
                        result_str += word + ' '
                        added = True
            else:
                if "PERSON" in entities:
                    if word in entities["PERSON"]:
                        #print "Word has tag PERSON"
                        result_str += word + ' '
        if result_str == "":
            sliding_window2(context, numsen, qid, question, pos, result)
        else:
            result[qid] = result_str


def sliding_window1(sentence, numofsen, qid, question, pos_dict, result):
    # Sliding Technique
    maxoverlap = -1 ##maximum of the number of overlap tokens
    maxnum = -1 ##the sentence with the most overlap
    startoverlap = -1 ##start position of the overlap
    endoverlap = -1 ##end position of the overlap
    qa_tokens=normalize(question).split()
    for j, x in enumerate(sentence): ##find the start and the end position of the overlap
        if x in qa_tokens:
            if startoverlap == -1:
                startoverlap = j
            endoverlap = j
    result[qid]=([])
    result_str=''
    ##use the part before the start of the overlap and the part after the end of the overlap as the answer to avoid too long answers
    ##and the answers are approximately arround the overlap
    for i,x in enumerate(sentence):
        #print(x)
        #if i<startoverlap:
        if x in pos_dict:
                #print(pos_dict[x])
            if 'NN'in pos_dict[x] or 'NNS'in pos_dict[x] or 'NNP'in pos_dict[x] or 'NNPS'in pos_dict[x]:
                result_str += x + ' '
                    #print('in sliding window: written')
        #if i>endoverlap:
        #    if x in pos_dict:
                #print(pos_dict[x])
        #        if 'NN'in pos_dict[x] or 'NNS'in pos_dict[x] or 'NNP'in pos_dict[x] or 'NNPS'in pos_dict[x]:
        #            result_str += x + ' '
                    #print('in sliding window: written')
    result[qid]=result_str

def sliding_window2(sentence, numofsen, qid, question, pos_dict, result):
    # Sliding Technique
    maxoverlap = -1 ##maximum of the number of overlap tokens
    maxnum = -1 ##the sentence with the most overlap
    startoverlap = -1 ##start position of the overlap
    endoverlap = -1 ##end position of the overlap
    qa_tokens=normalize(question).split()
    for j, x in enumerate(sentence): ##find the start and the end position of the overlap
        if x in qa_tokens:
            if startoverlap == -1:
                startoverlap = j
            endoverlap = j
    result[qid]=([])
    result_str=''
    ##use the part before the start of the overlap and the part after the end of the overlap as the answer to avoid too long answers
    ##and the answers are approximately arround the overlap
    for i,x in enumerate(sentence):
        #print(x)
        if i<startoverlap:
            result_str += x + ' '
                    #print('in sliding window: written')
        if i>endoverlap:
        #    if x in pos_dict:
                #print(pos_dict[x])
        #        if 'NN'in pos_dict[x] or 'NNS'in pos_dict[x] or 'NNP'in pos_dict[x] or 'NNPS'in pos_dict[x]:
            result_str += x + ' '
                    #print('in sliding window: written')
    result[qid]=result_str

def main():
    dev = load()
    dev_inner = dev["data"]
    result = {}
    classifier = V.classifier()

    for documents in dev_inner:
        for paragraph in documents['paragraphs']:
            pos_dict = {}
            word_dict = {}
            entities = {}
            entity_dict = {}
            para = normalize(paragraph['context'])
            tokens = para.encode("utf-8").split()
            #encodetoken=[]
            #for x in tokens:
            #    encodetoken.append(x.encode("utf-8"))
            tagged = nltk.pos_tag(nltk.word_tokenize(paragraph['context']))
            #tagged = nltk.pos_tag(tokens)
            named_entities = nltk.chunk.ne_chunk(tagged)
            tags = named_entities.pos()
            for t in tags:
                word = t[0][0]
                if t[0][1] not in pos_dict:
                    pos_dict[t[0][1]] = []
                    pos_dict[t[0][1]].append(t[0][0])
                elif t[0][0] not in pos_dict[t[0][1]]:
                    pos_dict[t[0][1]].append(t[0][0])
                if t[0][0] not in word_dict:
                    word_dict[t[0][0]] = []
                    word_dict[t[0][0]].append(t[0][1])
                elif t[0][1] not in word_dict[t[0][0]]:
                    word_dict[t[0][0]].append(t[0][1])
                if t[1] not in entities:
                    entities[t[1]] = []
                    entities[t[1]].append(t[0][0])
                elif t[0][0] not in entities[t[1]]:
                    entities[t[1]].append(t[0][0])
                if word not in entity_dict:
                    entity_dict[word] = []
                    entity_dict[word].append(t[1])
                elif t[1] not in entity_dict[word]:
                    entity_dict[word].append(t[1])

            # for e in entities:
            #     print e
            #     print entities[e]

            paragraph_tokens=paragraph['context'].split() ##get tokens from context
            numofsen=0  ##number of sentences
            sentence={}
            sentence[numofsen] = ([])
            for x in paragraph_tokens:
                sentence[numofsen].append(normalize(x))
                if '.'in x or ','in x or '?'in x or '!'in x or ';'in x or ':'in x:  ##if reach any punctuation, sign as the end of the sentence, or the sentence will be very long
                    numofsen+=1
                    sentence[numofsen] = ([])

            sentence_vectors = []
            for i in range(0,numofsen):
                sentence_vectors.append(classifier.create_avg_vector(sentence[i]))

            for qa in paragraph['qas']: ##for each question, compute the unigram overlap
                tag= "OTHER"
                question = qa['question'].lower()
                if "what time" in question or "which time" in question or "what year" in question or "which year" in question or "what century" in question or "which century" in question or "what month" in question or "which month" in question or "what decade" in question or "which decade" in question:
                    tag = "TIM"
                elif "what place" in question or "which place" in question or "what area" in question or "which area" in question or "what town" in question or "which town" in question or "what state" in question or  "which state" in question or "what city" in question or "which city" in question or "what country" in question or "which country" in question:
                    tag = "LOC"
                elif "what person" in question or "which person" in question:
                    tag = "PER"
                elif "what" in question or "which" in question:
                    tag = "OTHER1"
                #check for anytime what/how comes before when, tag as OTHER
                #check for "on what" tag as LOC
                #check for what followed by a LOC/PER/TIM tag
                #check for "what time" tag as TIM
                elif "where" in question:
                    tag = "LOC"
                elif "when" in question:
                    tag = "TIM"
                elif "who" in question:
                    #covers whom/whose as well
                    tag = "PER"
                else:
                    tag = "OTHER2"

                maxsim = -1 ##maximum of the similarity
                maxnum = 0 ##the sentence with the maximal similarity
                qa_tokens=normalize(qa['question']).split()
                qa_vector = classifier.create_avg_vector(qa_tokens)
                for i in range(0,numofsen):
                    total=0
                    total= classifier.cosine_similarity(sentence_vectors[i], qa_vector)##compute the similarity
                    # print total
                    if total>=maxsim: ##find the maximum similarity
                        #print ("in if", i, total)
                        maxsim=total
                        maxnum=i
                better_qa(tag, sentence[maxnum], numofsen, qa['id'], qa['question'], entities, word_dict, result)
        store(result) ##write the answers to json file

if __name__ == '__main__':
    main()