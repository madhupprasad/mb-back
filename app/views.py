from flask import Blueprint, json, jsonify, request
from elasticsearch import Elasticsearch

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
