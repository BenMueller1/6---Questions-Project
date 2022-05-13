import nltk
#  nltk.download('stopwords')   only have to run this once
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dict = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, encoding="utf8") as file:
            dict[filename] = file.read()
    return dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    finalwords = []
    for i in range(len(words)):
        words[i] = words[i].lower()
        if words[i] not in string.punctuation and words[i] not in nltk.corpus.stopwords.words("english"):
            if any(c.isalpha() for c in words[i]):
                finalwords.append(words[i])
    return finalwords


def get_all_unique_words(documents):
    # just need to compute idf for words we are seeing for first time
    words = set()
    for document in documents.values():
        for word in document:
            words.add(word)
    return list(words)

def compute_idf(word, documents):
    # idf is defined as log[(num docs)/(num docs containing word)]
    num_documents = len(list(documents.values()))
    num_docs_containing_word = 0
    for document in documents:
        if word in document:
            num_docs_containing_word += 1
    if num_docs_containing_word == 0:
        return 0   # IS THIS CORRECT? IS THE IDF ZERO WHEN ITS NOT PRESENT IN ANY DOC?
    return math.log2(num_documents/num_docs_containing_word)

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words_to_idfs = {}
    unique_words = get_all_unique_words(documents)
    for word in unique_words:
        words_to_idfs[word] = compute_idf(word, documents)
    return words_to_idfs


def find_top_n_scores(n, files_to_scores):
    top_n_filenames = []
    for _ in range(n):
        top = max(files_to_scores, key=files_to_scores.get)  # gets key with maximum value
        del files_to_scores[top]
        top_n_filenames.append(top)
    return top_n_filenames

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    files_to_scores = {} # maps each doc to a score representing how relevant it is to query
    for title, document in files.items():
        score = 0
        for word in query:
            score += document.count(word) * idfs[word]  # will add nothing if the word isnt present
        files_to_scores[title] = score

    # we must find and remove (and add to return list) the max from files_to_scores n times
    return find_top_n_scores(n, files_to_scores)


def get_query_term_density(query, sentence):
    query_words = 0
    for word in sentence:
        if word in query:
            query_words += 1
    return (query_words/len(sentence))


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Sentences should be ranked according to “matching word measure”: namely, 
    # the sum of IDF values for any word in the query that also appears in the sentence.
    sentence_and_score_and_QTD = []
    for sentence, words in sentences.items():
        score = 0
        query_term_density = get_query_term_density(query, words)
        for word in words:
            if word in query:
                score += idfs[word]
        sentence_and_score_and_QTD.append([sentence, score, query_term_density])

    # sorts by the scores then by the QTD if there is a tie
    sentence_and_score_and_QTD.sort(key = lambda triple: (triple[1], triple[2]), reverse=True)  
    breakpoint()
    # return the top n sentences based on this sorting
    top_n_sentences = []
    for i in range(n):
        top_n_sentences.append(sentence_and_score_and_QTD[i][0])
    return top_n_sentences

if __name__ == "__main__":
    main()
