from sentence_transformers import SentenceTransformer
import numpy as np

# Download from the 🤗 Hub
model = SentenceTransformer("sbintuitions/sarashina-embedding-v1-1b")
# Run inference
sentences = [
    '更級日記は、平安時代中期に菅原孝標女によって書かれた回想録です。'
]
embedding_vector  = model.encode(sentences)[0]

embedding_array = np.array(embedding_vector, dtype=np.float32)

np.save("data/sarashina_embedding.npy", embedding_array)
