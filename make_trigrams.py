import nltk
import json

def get_posts():
    with open("rank_phones.txt") as f:
        buf = []
        for l in f:
            if l.startswith("AABBCCDDEEZZXXAABBCCDDEEZZXXAABBCCDDEEZZXX"):
                yield nltk.trigrams(nltk.word_tokenize("".join(buf)))
                buf = []
                continue
            buf.append(l)


for post in get_posts():
    for tg in post:
        print(json.dumps(list(tg)))
