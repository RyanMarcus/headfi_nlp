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
    yield from tokenizer.tokenize(post)


def post_as_labeled_tokens(post):
    for token in post_as_tokens(post):
        marked = "//MARK\\\\" in token
        yield (token.replace("//MARK\\\\", ""), marked)

def post_as_tagged_labeled_tokens(post):
    labeled_tokens = list(post_as_labeled_tokens(post))
    tokens = [t[0] for t in labeled_tokens]
    tags = filter(lambda x: x[1] not in string.punctuation, nltk.pos_tag(tokens))
    yield from ((x[0][0], x[0][1], x[1]) for x in zip(tags, (t[1] for t in labeled_tokens)))

def post_as_ngrams(post, gram_size=3):
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

def ngrams_as_training_data(ngrams):
    previous = "False"
    for ngram in ngrams:
        vec = dict()

        for ew in enumerate(ngram[0]):
            c, w = ew
            vec['pos' + str(c)] = str(w[1])
            vec['len' + str(c)] = len(w[0])
            vec['typ' + str(c)] = type_of_string(w[0])

        
        vec['prev'] = previous
        previous = str(ngram[1])


        yield vec, ngram[1]
    
def file_as_vectors(filename, gram_size=3):
    #print("f1,l1,t1,f2,l2,t2,f3,l3,t3,prev,label")
    for post in file_as_posts(filename):
        yield from ngrams_as_training_data(post_as_ngrams(post, gram_size=gram_size))


        
class Corpus:
    def __init__(self, filename):
        self.filename = filename

    def train(self, gram_size=3, evaluate_model=True):
        training = []
        labels = []

        logging.debug("Extracting training data")
        
        # train on a subset of the grams
        for vec in itertools.islice(file_as_vectors(self.filename), 5000):
            training.append(vec[0])
            labels.append(vec[1])


        # convert dict to vectors
        training = DictVectorizer().fit_transform(training).toarray()

        
        logging.debug("Training data extracted")



        
        self.clf = RandomForestClassifier(n_estimators=50, n_jobs=2)
        #self.clf = DecisionTreeClassifier(min_samples_leaf=2)
        
        logging.info("Starting model training...")
        self.clf = self.clf.fit(training, labels)
        logging.info("Model training complete.")
        
        if not evaluate_model:
            return

        logging.debug("Starting model evaluation...")

        predictions = cross_validation.cross_val_predict(self.clf, training, y=labels, n_jobs=2, cv=3)

        precision = sklearn.metrics.precision_score(labels, predictions)
        recall = sklearn.metrics.recall_score(labels, predictions)
        confusion = sklearn.metrics.confusion_matrix(labels, predictions)
                                                         
        
        logging.info("Recall (finding true positives): %s", recall)
        logging.info("Precision (avoid false positives): %s", precision)
        logging.info("Confusion matrix: \n\n %s", confusion)
        logging.debug("Finished model evaluation.")
        logging.info("Saving model to disk...")
        from sklearn.externals import joblib
        joblib.dump(self.clf, 'models/lcd_model.pkl')
        logging.info("Saved model")

    def load_model(self):
        from sklearn.externals import joblib
        clf = joblib.load('models/lcd_model.pkl')
        
        
               



# this file has 1017980 samples (trigrams)
c = Corpus("lcd_impressions.txt")
c.train()

