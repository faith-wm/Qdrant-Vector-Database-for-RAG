import pandas as pd
import json
import re
import pronto
import networkx as nx
import uuid
import torch
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
from transformers import AutoTokenizer, AutoModel
from torch.nn import DataParallel
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['hidden_states'][-1]
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=3000, return_tensors="pt").to('cuda')
    with torch.no_grad():
        embs = model(**inputs, output_hidden_states=True)
        embedding = mean_pooling(embs, inputs["attention_mask"])
    return embedding.cpu().numpy()


def batch_embed_texts(texts, model, tokenizer, batch_size=100, max_length=3000):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True)
            batch_embeddings = mean_pooling(embs, inputs["attention_mask"])
        embeddings.extend(batch_embeddings.cpu().numpy())
    return embeddings



def create_and_search_db():
    # client = QdrantClient(path="/home/qdrant_vectordb")
    client = QdrantClient(location=":memory:")
    
    checkpoint = "/models/NV-Embed-v2/"
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True).cuda()
    for module_key, module in model._modules.items():
        model._modules[module_key] = DataParallel(module)

    df=pd.read_csv('HPterms.csv')

    print('starting embeddings ',  time.strftime("%Y%m%d_%H%M"))

    embeddings = model._do_encode([x for x in df['embedding_text']], batch_size=8, instruction="", max_length=32768, num_workers=2, return_numpy=False)
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)

    df['embedding']=[emb.tolist()for emb in norm_embeddings]

    ###### creating the database
    print('emebedding done, creating db ',  time.strftime("%Y%m%d_%H%M"))
    my_collection='terms_db'
     # # # Delete the collection if it exists
    client.delete_collection(my_collection)
    ## create collection
    client.create_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE, on_disk=False),
    quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
    # optimizers_config=models.OptimizersConfigDiff(default_segment_number=5,indexing_threshold=0)
    )

    points = []
    for idx, data in df.iterrows():
        # if idx%100==0:
        #     print(idx)
        points.append(PointStruct(id=str(uuid.uuid4()), vector=data['embedding'],
                      payload={'id': data['term_id'],
                               'text':data['Output'],}))
    client.upsert(collection_name=my_collection, points=points)
    print('creating done, search starting ', time.strftime("%Y%m%d_%H%M"))

    #Querying the the database
    search_result = [client.search(collection_name=my_collection, query_vector=i,limit=10, search_params=models.SearchParams(hnsw_ef=12, exact=False, quantization=models.QuantizationSearchParams(ignore=False, rescore=True,oversampling=2.0,))) for i in df['embedding']]
    results=[[result.payload.get('text', 'NA') for result in search] for search in search_result ]
    

   
    
    
