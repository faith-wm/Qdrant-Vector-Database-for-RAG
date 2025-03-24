import argparse
import pandas as pd
import time
import uuid
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['hidden_states'][-1]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def batch_embed_texts(texts, model, tokenizer, batch_size=100, max_length=3000):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to('cuda')
        with torch.no_grad():
            embs = model(**inputs, output_hidden_states=True)
            batch_embeddings = mean_pooling(embs, inputs["attention_mask"])
        embeddings.extend(batch_embeddings.cpu().numpy())
    return embeddings


def create_and_search_db(args):
    client = QdrantClient(location=":memory:")

    print('Loading model...')
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)

    for module_key, module in model._modules.items():
        model._modules[module_key] = DataParallel(module)

    df = pd.read_csv(args.csv_path)
    texts = df[args.embedding_column].tolist()

    print('Starting embeddings', time.strftime("%Y%m%d_%H%M"))

    embeddings = model._do_encode(
        texts,
        batch_size=args.batch_size,
        instruction="",
        max_length=32768,
        num_workers=2,
        return_numpy=False
    )
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    df['embedding'] = [emb.tolist() for emb in norm_embeddings]

    print('Embedding done, creating DB...', time.strftime("%Y%m%d_%H%M"))

    client.delete_collection(args.collection_name)

    client.create_collection(
        collection_name=args.collection_name,
        vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE, on_disk=False),
        quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=row['embedding'],
            payload={
                'id': row['term_id'],
                'text': row['Output']
            }
        ) for _, row in df.iterrows()
    ]

    client.upsert(collection_name=args.collection_name, points=points)
    print('DB creation done. Starting search...', time.strftime("%Y%m%d_%H%M"))

    search_result = [
        client.search(
            collection_name=args.collection_name,
            query_vector=vector,
            limit=10,
            search_params=models.SearchParams(
                hnsw_ef=12,
                exact=False,
                quantization=models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0)
            )
        )
        for vector in df['embedding']
    ]

    results = [
        [result.payload.get('text', 'NA') for result in search]
        for search in search_result
    ]

    print("Search complete. Sample result:")
    print(results[0][:3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed text and create/search Qdrant vector DB.")

    parser.add_argument("--model_path", type=str, default="/models/NV-Embed-v2/", help="Path to embedding model.")
    parser.add_argument("--csv_path", type=str, default="terms.csv", help="Path to CSV file with input data.")
    parser.add_argument("--embedding_column", type=str, default="embedding_text", help="Column containing text to embed.")
    parser.add_argument("--collection_name", type=str, default="terms_db", help="Qdrant collection name.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding.")

    args = parser.parse_args()
    create_and_search_db(args)
