from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup

import bs4
import cPickle as pkl
import string
import re


# if this crashes open up a python session and run nltk.download() and download
# the appropriate corpus/model from the pop up menu
cached_stop_words = stopwords.words('english')
wpt = WordPunctTokenizer()
wnl = WordNetLemmatizer()


def process_text(text):
    """removes stopwords and lemmatize"""
    text = filter(lambda x: x in string.printable, text)
    text = text.strip().lower().decode('utf-8')
    text = re.sub('\-+|\_+|\++', ' ', text)
    words = wpt.tokenize(text)
    words = filter(lambda x: (x not in string.punctuation) and
                             (x not in cached_stop_words) and
                             (re.search('[a-z]', x)),
                   words)
    return [wnl.lemmatize(word) for word in words]


def strip_urls(text):
    """deletes replaces urls with ' ' characters"""
    return re.sub(r"^https?\/\/.*[\r\n]*", " ", text, flags=re.MULTILINE)


def visible(element):
    """
    takes a BeautifulSoup element
    return true when text is not part of set list of html sections
    """
    if element.parent.name in ['style', 'script', '[document]', 'meta', 'img', 'href', 'footer']:
        return False
    return not isinstance(element, bs4.element.Comment)


def grab_content_html(html):
    """ Strips raw html of non-text and returns only visible content"""
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.findAll(text=True)
    visible_texts = filter(visible, text)
    return visible_texts


def make_features(data):
    corpus = [' '.join(v['processed_text']) for v in data.itervalues()]
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)


if __name__ == '__main__':
    # Note I stripped tab characters ('\t') before I dumped them in the tsv file.
    # ' '.join(sample_string.split('\t'))
    lines = open('data/sample_text_labeled.tsv', 'r').readlines()
    data = {}
    for i,line in enumerate(lines):
        tokens = line.split('\t')
        processed_text = process_text(tokens[0])
        data[i] = {'processed_text': processed_text,
                   'label': int(tokens[1].strip())}

    # open and check output with
    #with open('data/processed_text.pkl', 'r') as f:
    #    data = pkl.load(f)

    #print data[0]
