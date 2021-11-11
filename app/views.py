from flask import Blueprint, json, jsonify, request
from elasticsearch import Elasticsearch

import json
import re
import nltk
import pickle
import datetime
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy import log10
import numpy as np

index = dict()  # inverted_index
document_vectors = dict()  # document vectors for scoring
ndocs = 0  # number of documents in the collection
documents_file_path = "/articles.json"
index_file_path = "/articles_indexed.pickle"
stop_words = set(stopwords.words("english"))

# Posting list structure for inverted index
# Example : {docId : "1", frequency : "5", tf_idf: "2.999"}
class Posting:
    def __init__(self, docId, frequency, tf_idf=0):
        self.docId = docId
        self.frequency = frequency
        self.tf_idf = tf_idf

    def __repr__(self):
        return str(self.__dict__)


def tokenize(document):
    tokens = nltk.word_tokenize(document, language="english")
    return tokens


def remove_stopwords(tokens):
    processed_tokens = []
    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            processed_tokens.append(token)
    return processed_tokens


def stem_words(tokens):
    ps = PorterStemmer()
    lemmatized_tokens = []
    for token in tokens:
        lemmatized_tokens.append(ps.stem(token))
    return lemmatized_tokens


def process_query(query):
    clean_text = re.sub(r"[^\w\s]", "", query)
    tokens = tokenize(clean_text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return tokens


def generate_inverted_index(document):

    clean_text = re.sub(r"[^\w\s]", "", document["title"])
    clean_text = clean_text + re.sub(r"[^\w\s]", "", document["subtitle"])
    clean_text = clean_text + re.sub(r"[^\w\s]", "", document["content"])
    tokens = tokenize(clean_text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    posting_dict = dict()

    for token in tokens:
        token_frequency = posting_dict[token].frequency if token in posting_dict else 0
        posting_dict[token] = Posting(document["id"], token_frequency + 1)

    updated_index = {
        key: [posting] if key not in index else index[key] + [posting]
        for (key, posting) in posting_dict.items()
    }

    index.update(updated_index)


def calculate_tf_idf():
    for term in index.keys():
        df = len(index[term])
        idf = log10(ndocs / df)
        for doc in index[term]:
            doc.tf_idf = doc.frequency * idf


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))


def gen_vectors(query):

    i = 0
    n = len(query)
    query_vector = [0] * len(query)
    for term in query:
        posting_list = index[term]
        # constructing tf_idf for terms in query vector
        query_term_tf_idf = log10(ndocs / len(posting_list))
        query_vector[i] = query_term_tf_idf
        # constructing document vectors
        for entry in posting_list:
            if entry.docId in document_vectors.keys():
                document_vectors[entry.docId][i] = entry.tf_idf
            else:
                document_vectors[entry.docId] = [0] * n
                document_vectors[entry.docId][i] = entry.tf_idf
        i = i + 1
    return query_vector


def scoring(query_vector):
    score = dict()
    for docId in document_vectors.keys():
        score[docId] = cosine_similarity(query_vector, document_vectors[docId])
    print("Score sorting start", datetime.datetime.now())
    sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    print("Score sorting end", datetime.datetime.now())

    return sorted_score


def start_indexing(documents):

    # Generate Inverted Index
    doc_num = 1
    start_time = datetime.datetime.now()
    print("Indexing Started")
    for document in documents:
        generate_inverted_index(document)
        print(doc_num, " ")
        doc_num = doc_num + 1
    print(
        "Indexing Started @ ",
        start_time,
        "\n",
        "Indexing Completed @ ",
        datetime.datetime.now(),
        "\n",
    )
    start_time = datetime.datetime.now()
    calculate_tf_idf()
    print(
        "TF-IDF Calculation Started @ ",
        start_time,
        "\n",
        "TF-IDF Calculation Completed @",
        datetime.datetime.now(),
    )

    # Persist the Index File
    index_file = open(index_file_path, "wb")
    pickle.dump(index, index_file)
    index_file.close()



with open(documents_file_path, "r") as file:
    documents = json.load(file)

ndocs = len(documents)

# start_indexing(documents)

# Load inverted_index
with open(index_file_path, "rb") as index_file:
    index = pickle.load(index_file)

# Query and get scoring
processed_query = process_query("Bigil Vijay Nayanthara Jackie Shroff Atlee Kumar")
print("Processed Query ", processed_query)
query_vector = gen_vectors(processed_query)
scores = scoring(query_vector)
print(scores[0:10])


#apis 

client = Elasticsearch(hosts="http://elastic:elastic@20.106.130.95:9200")
main = Blueprint('main', __name__)
@main.route('/get' , methods=['POST'])
def abc():
    input_data = request.get_json()
    result = client.search(index = "movies", query={"match": { "title" : input_data['query'] }})
    all_hits = result['hits']['hits']
    return_array = []
    for num, doc in enumerate(all_hits):
        return_array.append(doc['_source'])
    return jsonify({'data' : return_array})
