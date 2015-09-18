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


""" Depricating and using method of the same name in train_classifier
def make_features(data):
    corpus = [' '.join(v['processed_text']) for v in data.itervalues()]
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
"""
def make_features(data):
    """
    creates a matrix of stories with each row representing a story,
    each column a word in the corpus, and each value the occurences.

    another vector represents the category of each story-matching with the columns.
    """
    corpus = [' '.join(v['processed_text']) for v in data.itervalues()]
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    y = [v['label'] for v in data.itervalues()]

    return X, y, vectorizer


def process_tsv(tsv):
    """Takes a string location of tsv file
    Col 1 (story)   | Col 2 (label)
    -------------------------------
    Story 1         | 1
    Story 2         | 0

    returns a dictionary with the format:
    {Story 1:{processed_text:['word1', 'word2', 'word1', 'word3'], label:1}
    Story 2:{processed_text:['word2', 'word2', 'word1', 'word5'], label:0}}
    """
    # Note I stripped tab characters ('\t') before I dumped them in the tsv file.
    lines = open(tsv, 'r').readlines()
    data = {}
    for i,line in enumerate(lines):
        tokens = line.split('\t')
        processed_text = process_text(tokens[0])
        data[i] = {'processed_text': processed_text,
                   'label': int(tokens[1].strip())}
    return data


def map_doc(story, existing_map):
    # get list of existing keywords, doc counts, word counts
    keword_map = existing_map['keyword_map']
    total_words = 0
    # w1:[docs w/ word, times word appears] for this doc only
    word_list = {}
    ## grab word
    next_word = 'placeholder' # replace with actual word in bucket
    total_words = total_words + 1
    prev = word_list.get(next_word)
    counts = [1, 1]
    if len(prev) == 2:
        counts[0] = counts[0] + prev[0]
        counts[1] = counts[1] + prev[1]
    word_list[next_word] = counts
    # add word list to rest of this category
    # 
    return total_words, word_freq, word_list


def vectorize(data):
    """
    Takes a dictionary from process_tsv and returns
    List of categories
    List with doc count of each category
    Dict of Dict of Lists:
        {VC:{keword map: [w1, w2,...], doc occurences: [5, 10,...],
            word occurences:[77, 23,...], total docs, total words},
        PE{keyword map:[], doc occ:[], word occ:[], total docs,
            total words},
        ...
        OT{keyword map:[all keywords from all categories], doc occ:[],
        word occ:[], total docs, total words}}}
    """
    category_map = {'VC':{'keyword_map':[]}, 'PE':{}, 'M&A':{}, 'OT':{}}
    vckm = []
    makm = []
    pekem = []
    otkm = []



if __name__ == '__main__':
    tsv_name = 'data/sample_text_labeled.tsv'
    # remove stopwords and lemmatize
    data = process_tsv(tsv_name)
    vectorize(data)


    # ' '.join(sample_string.split('\t'))

    # open and check output with
    #with open('data/processed_text.pkl', 'r') as f:
    #    data = pkl.load(f)

    #print data[0]
