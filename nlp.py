import nltk
import string
import itertools
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import sklearn

from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
import logging
import numpy
from sklearn.externals import joblib

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

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
                        curr_post = ""
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

def numeric_fingerprint(s):
    return any(c.isdigit() for c in s) 


def extract_suffix(s):
    if len(s) <= 1:
        return "none"

    return s[-2:]

import re
model_sig_pattern = re.compile('[A-Z]+[A-Z0-9]*-[0-9][A-Z0-9]*')
def is_model_sig(s):
    return model_sig_pattern.match(s) != None


numeric_plural_or_poss_pattern = re.compile("[A-Za-z0-9]+'?s")
def is_numeric_plural(s):
    return numeric_plural_or_poss_pattern.match(s) != None


companies = {"ableplanet", "akg", "ar", "audeze", "altec", "audiotechnica", "beats", "behringer", "beyerdynamic", "bk", "bose", "b&w", "califone", "cortex", "creative", "cta", "denon", "etymotic", "everglide", "focal", "fostex", "futuresonics", "ge", "gemini", "genius", "grado", "hifiman", "icetech", "ifrogz", "inland", "isymphony", "jvc", "jwin", "klipsch", "koss", "labtec", "logitech", "ltb", "m-audio", "marshall", "maxell", "monoprice", "monster", "nady", "numark", "otto", "panasonic", "paradigm", "phiaton", "philips", "pioneer", "polk", "psb", "psyko", "rca", "roland", "samson", "sennheiser", "sherwood", "shure", "skullcandy", "sol", "sony", "soul", "stanton", "stax", "targus", "tascam", "tdk", "ultrasone", "velodyne", "vestax", "vic", "v-moda", "yamaha", "zagg", "alo", "audioquest", "burson", "fiio", "grace", "oppo", "schitt", "centrance", "woo"}
def is_company(s):
    res = s.lower() in companies
    res = res or (s.lower().replace("'s", "") in companies)
    res = res or (s.lower()[:-1] in companies)
    return res


def ends_sentence(s):
    s = s.strip()
    return s.endswith(".") or s.endswith("!") or s.endswith(";")


keywords = {"amp", "amps", "amplifier", "dac", "dacs", "cable", "cables", "headphones", "headphone", "HP"}
def is_keyword(s):
    if s in keywords:
        return s[0:2]
    return "n"

words = set(nltk.corpus.words.words())
def is_english(s):
    return s in words


def has_demarc(s):
    demarcs = ["$", "#", "%"]
    for d in demarcs:
        if d in s:
            return True

    return False

def features_for_ngrams(ngrams, previous):
    vec = dict()
    #vec['prev'] = previous
    for ew in enumerate(ngrams):
            c, w = ew
            vec['pos'       + str(c)] = str(w[1])
            vec['len'       + str(c)] = len(w[0])
            vec['typ'       + str(c)] = type_of_string(w[0])
            vec['num'       + str(c)] = numeric_fingerprint(w[0])
            vec['model_sig' + str(c)] = is_model_sig(w[0])
            vec['num_plurl' + str(c)] = is_numeric_plural(w[0])
            vec['is_compny' + str(c)] = is_company(w[0])
            vec['sent_term' + str(c)] = ends_sentence(w[0])
            vec['is_keywrd' + str(c)] = is_keyword(w[0])
            vec['is_englsh' + str(c)] = is_english(w[0])
            vec['has_demrc' + str(c)] = has_demarc(w[0])

    return vec
            
            
def ngrams_as_training_data(ngrams):
    previous = False
    for ngram in ngrams:
        vec = features_for_ngrams(ngram[0], previous)
        previous = ngram[1]
        yield vec, ngram[1]


def ngrams_as_data(ngrams):
    for ngram in ngrams:
        vec = features_for_ngrams(ngram, "FILL THIS IN")
        yield vec

def post_to_vector(post, gram_size):
    return list(ngrams_as_training_data(post_as_labeled_ngrams(post, gram_size)))

        
def file_as_training_vectors(filename, gram_size):
    worker_count = 8
    with ProcessPoolExecutor(max_workers=worker_count) as exe:
        itr =  enumerate(file_as_posts(filename))
        futures = []
        for i in range(worker_count):
            try:
                idx, post = next(itr)
                futures.append(exe.submit(post_to_vector, post, gram_size))
            except StopIteration:
                break # we've submitted all the tasks already!



        while True:
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            futures = []
            for i in done:
                yield from i.result()
                try:
                    idx, post = next(itr)
                    futures.append(exe.submit(post_to_vector, post, gram_size))
                except StopIteration:
                    # just wait it out
                    pass

            futures.extend(not_done)
            if len(futures) == 0:
                return
                
            
        




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
    def __init__(self, filenames, gram_size):
        self.filename = filenames
        self.gram_size = gram_size

    def extract(self, evaluate_model=True):
        

        training = []
        labels = []

        logging.debug("Extracting training data")

        
        samples_to_acquire = 37897
        for training_file in self.filename:
            print(training_file)
            for vec in file_as_training_vectors(training_file, self.gram_size):
                training.append(vec[0])
                labels.append(vec[1])
                if len(training) % 1000 == 0:
                    logging.debug("Extraction progress: %s / %s (%s)" % (len(training), samples_to_acquire, (len(training) / samples_to_acquire)))


        logging.debug("Got %s training instances" % len(training))
            
        # convert dict to vectors
        self.vectorizer = DictVectorizer()
        training = self.vectorizer.fit_transform(training).toarray()

        logging.debug("Saving vectorizer...")
        joblib.dump(self.vectorizer, "models/lcd_vectorize.pkl")
        
        logging.debug("Vectorizer saved, training data extracted")

        self.training = training
        self.labels = labels
        joblib.dump(self.training, "models/training_data.pkl")
        joblib.dump(self.labels, "models/training_labels.pkl")

        logging.debug("Training data and labels saved")


    def parameter_search(self):
        training = self.training
        labels = self.labels
        num_features = len(training[0])

        
        clf = GradientBoostingClassifier()
        gs = sklearn.grid_search.GridSearchCV(clf,
                                 { "learning_rate": [0.1, 0.5, 1.0],
                                   "n_estimators": [50, 100, 150],
                                   "max_depth": [3, int(math.ceil(math.log2(num_features))), num_features - 1],
                                   "max_features": ["auto", None],
                                   "min_samples_split": [1, 2, 4],
                                   "min_samples_leaf": [1, 2] },
                                              scoring="f1", n_jobs=8, verbose=1)


        # last best:
        #max_features=None,
        #min_samples_split=1,
        #min_samples_leaf=1,
        #learning_rate=0.1,
        #max_depth=3,
        #n_estimators=150
        
        gs.fit(training, labels)

        print(gs.grid_scores_)
        print(gs.best_estimator_)
        print(gs.best_score_)
        print(gs.best_params_)
        

        

    def train(self, evaluate_model=True, feature_importance=True):
        
        training = self.training
        labels = self.labels
        num_features = len(training[0])

        logging.debug("Feature count: %s" % num_features)
        
        #self.clf = BaggingClassifier(DecisionTreeClassifier(max_depth=num_features - 1,
        #                                                    min_samples_leaf=1,
        #                                                    min_samples_split=2,
        #                                                    max_features="sqrt",
        #                                                    class_weight="auto"),
        #                             n_jobs=1, n_estimators=500)
                                     
        self.clf = RandomForestClassifier(n_estimators=100, class_weight=None)

        #self.clf = GaussianNB()
        #self.clf = GradientBoostingClassifier(max_features=None,
        #                                      min_samples_split=1,
        #                                      min_samples_leaf=1,
        #                                      learning_rate=0.1,
        #                                      max_depth=3,
        #                                      n_estimators=150)
        #self.clf = KNeighborsClassifier(algorithm="auto")
        #self.clf = DecisionTreeClassifier(min_samples_leaf=2)

        if evaluate_model:
            logging.debug("Starting model evaluation...")

            predictions = cross_validation.cross_val_predict(self.clf, training, y=labels, n_jobs=-1, cv=3)

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

        if feature_importance:

            impor = dict()
            for k,v in self.vectorizer.vocabulary_.items():
                impor[k] = self.clf.feature_importances_[v]

            for k in sorted(impor, key=lambda x: impor[x]):
                logging.debug("Feature %s\thas importance\t%s" % (k, impor[k]))


        
        
        logging.info("Saving model to disk...")
        joblib.dump(self.clf, 'models/lcd_model.pkl')
        logging.info("Saved model")

    def load_model(self):
        self.clf = joblib.load('models/lcd_model.pkl')
        self.vectorizer = joblib.load('models/lcd_vectorize.pkl')

    def load_training_data(self):
        self.training = joblib.load('models/training_data.pkl')
        self.labels = joblib.load('models/training_labels.pkl')
        self.vectorizer = joblib.load('models/lcd_vectorize.pkl')
        
    def identify_entities(self, filename):
        if not hasattr(self, 'clf'):
            raise ValueError("Cannot identify elements without first loading or training a model")
            
        previous = False
        for vec, gram in file_as_vectors(filename, self.gram_size):

            vec['prev'] = previous
            sample = self.vectorizer.transform(vec).toarray()
            result = self.clf.predict(sample)
            previous = result[0]
            if result[0]:
                print(" ".join(gram), "|")
        
        
               

# with trigrams and CV search:
#INFO:root:Recall (finding true positives): 0.515901060071
#INFO:root:Precision (avoid false positives): 0.696897374702

                
import sys
if __name__ == "__main__":
    c = Corpus(["lcd_impressions.txt", "he400_impressions.txt", "asgard2_impressions.txt"], 3)
    if sys.argv[1] == "extract":
        c.extract()
    elif sys.argv[1] == "train":
        c.load_training_data()
        c.train()
    elif sys.argv[1] == "search":
        c.load_training_data()
        c.parameter_search()
    elif sys.argv[1] == "test":
        c.load_model()
        c.identify_entities("he400_nolabels.txt")
        
# this file has 1017980 samples (trigrams)

#c.extract()

#c.load_training_data()
#c.train()


#c.load_model()
#c.identify_entities("lcd_eval.txt")


#print(max(map(lambda x: x[0], enumerate(file_as_posts("lcd_impressions.txt")))))
    
