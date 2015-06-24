import urllib.request
from bs4 import BeautifulSoup
import re



class Post:
    def __init__(self, thread_id, post_id):
        self.thread_id = thread_id
        self.post_id = post_id
        self.post_refs = []

    def get_thread_id(self):
        return int(self.thread_id)

    def get_post_id(self):
        return int(self.post_id)

    def add_post_ref(self, ref):
        self.post_refs.append(int(ref))

    def get_post_refs(self):
        return self.post_refs
    
    def __repr__(self):
        return str(self.get_thread_id()) + "/" + str(self.get_post_id()) + " references " + str(self.get_post_refs())

post_id_pattern = re.compile('content_([0-9]+)')
post_ref_pattern = re.compile('.+post_([0-9]+)')

def build_url(thread_id, post_num):
    toR = "http://head-fi.org/t/" + str(thread_id)
    if post_num != None:
        toR += "/xxx/" + str(post_num)

    print("Built:", toR)
        
    return toR

def scanThreadPage(thread_id, post_num=None):
    """ thread_id is the numerical ID from the url, ex 588429 for LCD-3 impressions """
    
    t = urllib.request.urlopen(build_url(thread_id, post_num))
    soup = BeautifulSoup(t.read())
    posts = soup.find_all("div", class_="post-content-area")
    toR = []
    for x in posts:
        post_id = int(post_id_pattern.match(x.find_all("div", class_="shazam")[0]['id']).group(1))

        toR = Post(thread_id, post_id)

        post_refs = [int(post_ref_pattern.match(y.a['href']).group(1)) for y in x.find_all("div", class_="quote-block")]
        for x in post_refs:
            toR.add_post_ref(x)

        yield toR


def scanThread(thread_id):
    yield from scanThreadPage(thread_id)

    c = 15
    while True:
        capt = list(scanThreadPage(thread_id, c))
        if len(capt) == 0:
            return
        for p in capt:
            yield p
        c += 15
        
        

        

for post in scanThread(588429):
    print(post)
