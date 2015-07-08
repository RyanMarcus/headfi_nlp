import nltk
import string
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import sklearn

from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
import logging
import numpy
from sklearn.externals import joblib

logging.basicConfig(level=logging.DEBUG)

def file_as_posts(filename):
    with open(filename, "r") as f:
        curr_post = ""
        first_div = False
        for l in f:
            if l.startswith("//STOP\\\\"):
                if curr_post != "":
                    yield curr_post
                return

            if l.startswith("============================================="):
                if not first_div:
                    first_div = True
                    if curr_post != "":
                        yield curr_post
                    continue
                else:
                    first_div = False
                continue
            
            if l.startswith("Topic:") and first_div:
                continue

            curr_post += l

       

def post_as_tokens(post):
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for t in tokenizer.tokenize(post):
        yield t.replace("//MARK\\\\", "")


def post_as_labeled_tokens(post):
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    for token in tokenizer.tokenize(post):
        marked = "//MARK\\\\" in token
        yield (token.replace("//MARK\\\\", ""), marked)

def post_as_tagged_labeled_tokens(post):
    labeled_tokens = list(post_as_labeled_tokens(post))
    tokens = [t[0] for t in labeled_tokens]
    tags = filter(lambda x: x[1] not in string.punctuation, nltk.pos_tag(tokens))
    yield from ((x[0][0], x[0][1], x[1]) for x in zip(tags, (t[1] for t in labeled_tokens)))


def post_as_tagged_tokens(post):
    tokens = list(post_as_tokens(post))
    tags = filter(lambda x: x[1] not in string.punctuation, nltk.pos_tag(tokens))
    yield from tags


def post_as_ngrams(post, gram_size):
    last_n_tokens = []
 
    def insert_token(t):
        last_n_tokens.append(t)
        if len(last_n_tokens) > gram_size:
            last_n_tokens.pop(0)
    
    for i in post_as_tagged_tokens(post):
        insert_token(i)
        if len(last_n_tokens) == gram_size:
            yield list(last_n_tokens)
    
    
def post_as_labeled_ngrams(post, gram_size):
    last_n_tokens = []
    marked = []
    center = int(gram_size / 2)
 
    def insert_token(t):
        last_n_tokens.append((t[0], t[1]))
        marked.append(t[2])
        if len(last_n_tokens) > gram_size:
            last_n_tokens.pop(0)
            marked.pop(0)
    
    for i in post_as_tagged_labeled_tokens(post):
        insert_token(i)
        if len(last_n_tokens) == gram_size:
            yield (last_n_tokens, marked[center])
           

def type_of_string(s):

    if s.istitle():
        return "title"

    if s.isupper():
        return "upper"

    if s.islower():
        return "lower"

    if s.isalpha():
        return "alpha" # really mixed case

    if s.isdigit():
        return "num"

    if s.isalnum():
        return "alnum"

    return "none"


def features_for_ngrams(ngrams, previous):
    vec = dict()
    vec['prev'] = previous
    for ew in enumerate(ngrams):
            c, w = ew
            vec['pos' + str(c)] = str(w[1])
            vec['len' + str(c)] = len(w[0])
            vec['typ' + str(c)] = type_of_string(w[0])

    return vec
            
            
def ngrams_as_training_data(ngrams):
    previous = "False"
    for ngram in ngrams:
        vec = features_for_ngrams(ngram[0], previous)
        previous = str(ngram[1])
        yield vec, ngram[1]


def ngrams_as_data(ngrams):
    for ngram in ngrams:
        vec = features_for_ngrams(ngram, "FILL THIS IN")
        yield vec

    
def file_as_training_vectors(filename, gram_size):
    #print("f1,l1,t1,f2,l2,t2,f3,l3,t3,prev,label")
    for post in file_as_posts(filename):
        yield from ngrams_as_training_data(post_as_labeled_ngrams(post, gram_size))


def first_of_each(item):
    toR = []
    for i in item:
        toR.append(i[0])

    return tuple(toR)
        
def file_as_vectors(filename, gram_size):
    for post in file_as_posts(filename):
        grams = list(post_as_ngrams(post, gram_size))
        data = list(ngrams_as_data(grams))
        yield from zip(data, map(first_of_each, grams))
    

        
class Corpus:
    def __init__(self, filename, gram_size):
        self.filename = filename
        self.gram_size = gram_size

    def train(self, evaluate_model=True):
        

        training = []
        labels = []

        logging.debug("Extracting training data")
        
        # train on a subset of the grams
        for vec in itertools.islice(file_as_training_vectors(self.filename, self.gram_size), 500000):
            training.append(vec[0])
            labels.append(vec[1])


        # convert dict to vectors
        self.vectorizer = DictVectorizer()
        training = self.vectorizer.fit_transform(training).toarray()

        logging.debug("Saving vectorizer...")
        joblib.dump(self.vectorizer, "models/lcd_vectorize.pkl")
        
        logging.debug("Vectorizer saved, training data extracted")



        
        self.clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
        #self.clf = DecisionTreeClassifier(min_samples_leaf=2)

        if evaluate_model:
            logging.debug("Starting model evaluation...")

            predictions = cross_validation.cross_val_predict(self.clf, training, y=labels, n_jobs=2, cv=3)

            precision = sklearn.metrics.precision_score(labels, predictions)
            recall = sklearn.metrics.recall_score(labels, predictions)
            confusion = sklearn.metrics.confusion_matrix(labels, predictions)
                                                         
        
            logging.info("Recall (finding true positives): %s", recall)
            logging.info("Precision (avoid false positives): %s", precision)
            logging.info("Confusion matrix: \n\n %s", confusion)
            logging.debug("Finished model evaluation.")
        
        logging.info("Starting model training...")
        self.clf = self.clf.fit(training, labels)
        logging.info("Model training complete.")
        
        
        logging.info("Saving model to disk...")
        joblib.dump(self.clf, 'models/lcd_model.pkl')
        logging.info("Saved model")

    def load_model(self):
        self.clf = joblib.load('models/lcd_model.pkl')
        self.vectorizer = joblib.load('models/lcd_vectorize.pkl')

    def identify_entities(self, filename):
        if not hasattr(self, 'clf'):
            raise ValueError("Cannot identify elements without first loading or training a model")
            
        previous = "False"
        for vec, gram in file_as_vectors(filename, self.gram_size):

            vec['prev'] = previous
            sample = self.vectorizer.transform(vec)
            result = self.clf.predict(sample)
            previous = str(result[0])
            if result[0]:
                print(gram, result)
        
        
               



# this file has 1017980 samples (trigrams)
c = Corpus("lcd_impressions.txt", 3)
c.train()


#c.load_model()
c.identify_entities("lcd_eval.txt")

