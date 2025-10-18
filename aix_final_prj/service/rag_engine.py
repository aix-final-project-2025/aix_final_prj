import os
import fitz
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 임베딩 모델 및 LLM 초기화
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "skt/kogpt2-base-v2"

embedder = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def build_vector_db(chunks, index_path="faiss_index"):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, index_path)
    np.save("chunks.npy", chunks)

def search_similar_chunks(query, index_path="faiss_index", k=3):
    index = faiss.read_index(index_path)
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    chunks = np.load("chunks.npy", allow_pickle=True)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_answer(query, context_chunks):
    if not context_chunks:
        return "No relevant context found."
    context = "\n".join(context_chunks)
    prompt = f"다음 문서를 기반으로 질문에 답하세요:\n{context}\n\n질문: {query}\n답변:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            num_return_sequences=1,
            #temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("답변:")[-1].strip()