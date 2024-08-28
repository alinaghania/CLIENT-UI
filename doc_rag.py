import cohere

# Configure the Cohere Bedrock client
co = cohere.BedrockClient(
    aws_region="us-east-1",
    aws_access_key="your_access_key",
    aws_secret_key="your_secret_key",
    aws_session_token="your_session_token",
)

def generate_embedding(query):
    # Invoke Cohere embedding model via Bedrock
    texts = [query]
    input_type = "clustering"
    truncate = "NONE"  # optional
    model_id = "cohere.embed-english-v3"  # or "cohere.embed-multilingual-v3"

    result = co.embed(
        model=model_id,
        input_type=input_type,
        texts=texts,
        truncate=truncate
    )
    
    embedding = result.embeddings[0]  # Assuming only one text input
    return embedding

def search_similar_documents(embedding):
    from opensearchpy import OpenSearch
    
    client = OpenSearch(
        hosts=[{'host': 'your-opensearch-domain', 'port': 443}],
        http_auth=('your-username', 'your-password'),
        use_ssl=True
    )
    
    query = {
        "size": 5,
        "query": {
            "knn": {
                "field_name": "embedding_vector",
                "query_vector": embedding,
                "k": 5
            }
        }
    }
    
    response = client.search(body=query, index="your-index-name")
    documents = [hit['_source']['document'] for hit in response['hits']['hits']]
    
    return documents
