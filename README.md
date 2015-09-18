#PBClassifiers
Software to determine keywords differentiating news articles within categories.

##TODO
1. Grab training set from dbd_copy
* ~~grab_content_html~~
* ~~strip_urls~~
* ~~Strip tab characters from every story.~~
* generate tsv of stories-ending in 1 or 0
* ~~process_text on tsv~~
* vectorizer
  - convert all docs to a vector of frequencies where the vector is comprised of all non single-occurrence words in the corpus.
  - **get ratio of occurrences of the keyword in that category vs the other (accuracy of that keyword of capturing docs in that category)**
  - get **ratio of docs in that category that have that term (strength of keyword to capture that category)**
  - **extra** get weight of that keyword in the document (tf)
  - get uniqueness of that word in all documents (idf)
* print ordered list of keywords with inter-cat freq * intra-cat freq
*
