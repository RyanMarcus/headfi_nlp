import urllib.request
from bs4 import BeautifulSoup
import re
import networkx as nx
import nltk
import json


class Post:
    def __init__(self, thread_id, post_id, html):
        self.thread_id = thread_id
        self.post_id = post_id
        self.post_refs = []

        soup = BeautifulSoup(str(html))
        
        self.html = str(soup)
        self.prev = None

        # remove anything inside of a div with class quote-block
        for quoted in soup.find_all("div", class_="quote-container"):
            quoted.clear()

        # remove anything inside of a div with class forum-post-tools
        for tools in soup.find_all("div", class_="forum-post-tools"):
            tools.clear()

        # remove anything inside of a script tag
        for scripts in soup.find_all("script"):
            scripts.clear()
            
        self.text = soup.get_text()


        
        

    def get_thread_id(self):
        return int(self.thread_id)

    def get_post_id(self):
        return int(self.post_id)

    def get_html(self):
        return self.html
    
    def add_post_ref(self, ref):
        self.post_refs.append(int(ref))

    def get_post_refs(self):
        return self.post_refs

    def get_previous(self):
        return self.prev

    def set_previous(self, prev):
        self.prev = prev

    def get_text(self):
        return self.text

    def is_long_post(self):
         tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(self.get_text())
         return len(tokens) > 150
        
    def __repr__(self):
        return str(self.get_thread_id()) + "/" + str(self.get_post_id()) + " references " + str(self.get_post_refs())


class PostGraph:
    def __init__(self, posts):
        self.g = nx.DiGraph()
        all_posts = list(posts)
        for post in all_posts:
            self.g.add_node(post.get_post_id())

        for post in all_posts:
            for ref in post.get_post_refs():
                self.g.add_edge(post.get_post_id(), ref)


        self.post_dict = dict()
        for post in all_posts:
            self.post_dict[post.get_post_id()] = post
                

                
    def print_graphviz(self):
        print("digraph headfi {")
        print("rankdir=LR;")
        for e in self.g.edges_iter():
            print (e[0], "->", e[1], ";")
        print("}")

    def get_all_topics(self):
        return nx.weakly_connected_components(self.g)

    def get_all_posts(self):
        for v in self.post_dict.values():
            yield v
    
    def get_long_posts(self):
       
        return map(lambda x: x.get_post_id(), filter(is_long_post, self.get_all_posts()))


    def get_post_tokens(self, post_id):
        post = self.post_dict[post_id]
        return nltk.word_tokenize(post.get_text())

    def aggrogate_text(self, nodes):
        toR = ""
        for n in nodes:
            toR += self.post_dict[n].get_text()

        return toR

    def topics(self, nodes):
        text = self.aggrogate_text(nodes)
        bigrams = nltk.collocations.BigramAssocMeasures()
        trigrams = nltk.collocations.TrigramAssocMeasures()

        lemmatizer = nltk.WordNetLemmatizer()

        text = ''.join(filter(lambda x: x not in [">", "<", "-", "+"], text))
        words = nltk.word_tokenize(text)

        words = filter(lambda x: x not in nltk.corpus.stopwords.words('english'), words)
        words = map(lambda x: lemmatizer.lemmatize(x), words)
        
        finder = nltk.collocations.BigramCollocationFinder.from_words(words)
        finder.apply_freq_filter(2)
        return finder.nbest(bigrams.pmi, 10)


post_id_pattern = re.compile('content_([0-9]+)')
post_ref_pattern = re.compile('.+post_([0-9]+)')


def backwards_link(func):
    def toR(*args, **argk):
        last = None
        for i in func(*args, **argk):
            i.set_previous(last)
            last = i
            yield i

    return toR


def cache_to_file(file_name):
    def decorator(original_func):
        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}

        def new_func(*param):
            key = str(param)
            if key not in cache:
                cache[key] = original_func(*param)
                json.dump(cache, open(file_name, 'w'))

            return cache[key]

        return new_func

    return decorator


@cache_to_file('headfi-http-cache.dat')
def get_url(thread_id, post_num):
    return urllib.request.urlopen(build_url(thread_id, post_num)).read().decode('utf-8')

def build_url(thread_id, post_num):
    toR = "http://head-fi.org/t/" + str(thread_id)
    if post_num != None:
        toR += "/xxx/" + str(post_num)

        
    return toR

def scan_thread_page(thread_id, post_num=None):
    """ thread_id is the numerical ID from the url, ex 588429 for LCD-3 impressions """
    

    soup = BeautifulSoup(get_url(thread_id, post_num))
    posts = soup.find_all("div", class_="post-content-area")
    for x in posts:
        post_id = int(post_id_pattern.match(x.find_all("div", class_="shazam")[0]['id']).group(1))

        toR = Post(thread_id, post_id, x)

        post_refs = []
        try:
            post_refs = [int(post_ref_pattern.match(y.a['href']).group(1)) for y in x.find_all("div", class_="quote-block")]
        except:
            pass # no matches in regex
        
        for x in post_refs:
            toR.add_post_ref(x)

        yield toR

@backwards_link
def scan_thread(thread_id, max_pages=None):
    yield from scan_thread_page(thread_id)

    c = 15
    while True:
        capt = list(scan_thread_page(thread_id, c))
        if len(capt) == 0:
            return
        for p in capt:
            yield p
        c += 15

        if max_pages and c/15 >= max_pages:
            break
        
        

        

#pg = PostGraph(scan_thread(588429))

for topic in filter(lambda x: x.is_long_post(), scan_thread(588429)):
    print("=============================================")
    print("Topic:", topic.get_post_id(), "in thread", topic.get_thread_id())
    print("=============================================")
    print(topic.get_text())
        

        
