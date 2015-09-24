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
import pprint

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
    Create a matrix of stories with each row representing a story,
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
    {Story 1:{processed_text:['word1', 'word2', 'word1', 'word3'], label:1},
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


def map_story(processed_text):
    """Count occ of each word in story and total words"""
    total_words = 0
    # create new map of words in story
    # {'w1':occ, 'w2':occ,..., 'wn':occ}
    # where occ = times word appears for this doc only
    story_word_list = {}
    # grab words from story and increment counts of occ accordingly
    for word in processed_text:
        total_words = total_words + 1
        word_occ = 1
        # add count to existing if word already occured in story
        if word in story_word_list:
            word_occ = story_word_list.get(word) + word_occ
        # map new/updated count of occ to story_word_list
        story_word_list[word] = word_occ
        temp_output.write('found %i occurences of %r; %i total words\n' % (word_occ, word, total_words))
    return total_words, story_word_list


def update_cat_counts(total_words, story_word_list, cat_map):
    """
    Take parsed story info and updates the coresponding category map
    story_total = total words from story
    story_word_list = dic{'w1':occ of 1, 'w2':occ of 2,...}
    cat_map = {'keyword_map':{'w1':[occ, docs]}, 'total_docs':5, 'total_words':200}
    """
    # update total word count, and total docs in existing_map
    cat_map['total_words'] = cat_map['total_words'] + total_words
    cat_map['total_docs'] = cat_map['total_docs'] + 1
    for word in story_word_list:
        # counts = [total occ, total docs]
        counts = [story_word_list[word], 1]
        existing_word_list = cat_map['keyword_map']
        # add to other occ if word found in prev story
        if word in existing_word_list:
            prev_counts = existing_word_list[word]
            counts[0] = counts[0] + prev_counts[0]
            counts[1] = counts[1] + prev_counts[1]
        existing_word_list[word] = counts
        cat_map['keyword_map'] = existing_word_list
    return cat_map


def map_data(data):
    """ **PROBABLY A WORSE IMPLEMENTATION OF make_features()**
    Takes a dictionary from process_tsv and returns
    Dict of Dict of Lists:
        {VC:{keword map:{w1:[occ, docs] w2:[occ, docs],...]}, total docs,
            total words},
        PE:{keyword map:{}, total docs, total words}
        ...
        OT:{keyword map:{all keywords from all categories}, total docs,
            total words} }
    TODO: handle multipe maps in this method rather than update_cat_map
    """
    category_map = {}
    # label->category map: story labels should follow this indexing scheme
    # or you can just label with strings and remove this
    categories = ['OT', 'VC', 'PE', 'MA'] #removed OT to fix maps_impacted issue
    for cat in categories:
        category_map[cat] = {'keyword_map':{}, 'total_words':0, 'total_docs':0}
    for story in data:
        story_obj = data[story]
        processed_text = story_obj['processed_text']
        total_words, story_word_list = map_story(processed_text)
        # grab correct story category based on story label (stories[label] = 1)
        story_cat = categories[story_obj['label']] # type(label) == string
        story_cat_map = category_map[story_cat]
        ot_map = category_map['OT']
        # update both categories if this story is not OT
        if story_cat_map is not ot_map:
            print 'processing %r words from a %r story' % (total_words, story_cat)
            update_cat_counts(total_words, story_word_list, story_cat_map)
        update_cat_counts(total_words, story_word_list, ot_map)
    return category_map


if __name__ == '__main__':
    # TODO: remove test prints below
    temp_output = open('temp_output.txt', 'a')
    temp_output.truncate()

    tsv_name = 'data/sample_text_labeled.tsv'
    # remove stopwords and lemmatize
    data = process_tsv(tsv_name)
    category_map = map_data(data)
    output = open('TestOutput.txt', 'w')
    pprint.pprint(data, output)
    pprint.pprint(category_map, output)


    # ' '.join(sample_string.split('\t'))

    # open and check output with
    #with open('data/processed_text.pkl', 'r') as f:
    #    data = pkl.load(f)

    #print data[0]
