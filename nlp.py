import nltk
import string


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
        print(t)
        last_n_tokens.append((t[0], t[1]))
        marked.append(t[2])
        if len(last_n_tokens) > gram_size:
            last_n_tokens.pop(0)
            marked.pop(0)
    
    for i in post_as_tagged_labeled_tokens(post):
        insert_token(i)
        if len(last_n_tokens) == gram_size:
            yield (last_n_tokens, marked[center])
           

def ngrams_as_training_data(ngrams):
    for ngram in ngrams:
        vec = []
        for w in ngram[0]:
            vec.append(str(w[1]))

        vec.append(str(ngram[1]))
        yield vec
    

class Corpus:
    def __init__(self, filename):
        self.filename = filename

    def parse(self, gram_size=3):
        print("f1,f2,f3,label")
        for post in file_as_posts(self.filename):
            for tvec in ngrams_as_training_data(post_as_ngrams(post)):
                print(",".join(tvec))
               



c = Corpus("lcd_impressions.txt")
c.parse()
