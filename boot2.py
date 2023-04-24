import os

# Setup Redis
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

import tiktoken
import textract

from kbot.database import get_redis_connection

# The transformers.py file contains all of the transforming functions, including ones to chunk, embed and load data
# For more details the file and work through each function individually
from kbot.transformers import handle_file_string

redis_client = get_redis_connection()

# Constants
VECTOR_DIM = 1536 #len(data['title_vector'][0]) # length of the vectors
#VECTOR_NUMBER = len(data)                 # initial number of vectors
PREFIX = "sportsdoc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)

# Create search index

# Index
INDEX_NAME = "f1-index"           # name of the search index
VECTOR_FIELD_NAME = 'content_vector'

# Define RediSearch fields for each of the columns in the dataset
# This is where you should add any additional metadata you want to capture
filename = TextField("filename")
text_chunk = TextField("text_chunk")
file_chunk_index = NumericField("file_chunk_index")

# define RediSearch vector fields to use HNSW index

text_embedding = VectorField(VECTOR_FIELD_NAME,
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC
    }
)
# Add all our field objects to a list to be created as an index
fields = [filename,text_chunk,file_chunk_index,text_embedding]

# Is redis online?
redis_client.ping()

# Optional step to drop the index if it already exists
#redis_client.ft(INDEX_NAME).dropindex()

# Check if index exists
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except Exception as e:
    print(e)
    # Create RediSearch Index
    print('Not there yet. Creating')
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )

data_dir = os.path.join(os.curdir,'data')
pdf_files = sorted([x for x in os.listdir(data_dir) if 'DS_Store' not in x])

# This step takes about 5 minutes

# Initialise tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Process each PDF file and prepare for embedding
for pdf_file in pdf_files:
    
    pdf_path = os.path.join(data_dir, pdf_file)
    print(pdf_path)
    
    # Extract the raw text from each PDF using textract
    text = textract.process(pdf_path, method='pdfminer')
    print(text)
    # Chunk each document, embed the contents and load to Redis
    handle_file_string((pdf_file,text.decode("utf-8")),tokenizer,redis_client,VECTOR_FIELD_NAME,INDEX_NAME)


# Check that our docs have been inserted
redis_client.ft(INDEX_NAME).info()['num_docs']