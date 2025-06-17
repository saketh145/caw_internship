"""
Simple in-memory vector store as an alternative to ChromaDB.
This is a temporary solution for testing when ChromaDB is not available.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import os


class SimpleVectorStore:
    """Simple in-memory vector store for document embeddings."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.document_ids = []
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_file = os.path.join(persist_directory, f"{collection_name}.json")
        # Load existing data if available
        self.load()
    
    def add_embeddings(self, embeddings: List[List[float]], documents: List[str], 
                      metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add embeddings and documents to the store."""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        if ids is None:
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
        
        # Convert embeddings to numpy arrays for easier computation
        embeddings_array = np.array(embeddings)
        
        self.embeddings.extend(embeddings_array)
        self.documents.extend(documents)
        self.metadata.extend(metadatas)
        self.document_ids.extend(ids)
        
        # Auto-save
        self.save()
    
    def add(self, embeddings: List[List[float]], documents: List[str], 
            metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add embeddings and documents to the store (ChromaDB compatible interface)."""
        return self.add_embeddings(embeddings, documents, metadatas, ids)
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5, where: Dict[str, Any] = None) -> Dict[str, List]:
        """Query the vector store for similar documents."""
        if not self.embeddings:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'ids': [[]]
            }
        
        # Filter by metadata if where clause is provided
        valid_indices = list(range(len(self.documents)))
        if where:
            valid_indices = []
            for i, meta in enumerate(self.metadata):
                match = True
                for key, value in where.items():
                    if key not in meta or meta[key] != value:
                        match = False
                        break
                if match:
                    valid_indices.append(i)
        
        if not valid_indices:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'ids': [[]]
            }
        
        query_array = np.array(query_embeddings[0])  # Take first query
        
        # Calculate cosine similarity for valid indices only
        similarities = []
        for i in valid_indices:
            embedding = np.array(self.embeddings[i])
            # Cosine similarity
            dot_product = np.dot(query_array, embedding)
            norm_a = np.linalg.norm(query_array)
            norm_b = np.linalg.norm(embedding)
            similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            similarities.append((1 - similarity, i))  # Convert to distance (lower is better)
        
        # Sort by similarity and get top n_results
        similarities.sort(key=lambda x: x[0])
        top_n = min(n_results, len(similarities))
        top_similarities = similarities[:top_n]
        
        result_docs = [self.documents[idx] for _, idx in top_similarities]
        result_metadata = [self.metadata[idx] for _, idx in top_similarities]
        result_distances = [dist for dist, _ in top_similarities]
        result_ids = [self.document_ids[idx] for _, idx in top_similarities]
        
        return {
            'documents': [result_docs],
            'metadatas': [result_metadata],
            'distances': [result_distances],
            'ids': [result_ids]
        }
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self.documents)
    
    def get(self, ids: List[str] = None, where: Dict[str, Any] = None, limit: int = None) -> Dict[str, List]:
        """Get documents from the store (ChromaDB compatible interface)."""
        if not self.documents:
            return {
                'documents': [],
                'metadatas': [],
                'ids': []
            }
        
        # Filter by IDs if provided
        valid_indices = list(range(len(self.documents)))
        if ids:
            valid_indices = [i for i, doc_id in enumerate(self.document_ids) if doc_id in ids]
        
        # Filter by metadata if where clause is provided
        if where:
            filtered_indices = []
            for i in valid_indices:
                meta = self.metadata[i]
                match = True
                for key, value in where.items():
                    if key not in meta or meta[key] != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
            valid_indices = filtered_indices
        
        # Apply limit if provided
        if limit and len(valid_indices) > limit:
            valid_indices = valid_indices[:limit]
        
        result_docs = [self.documents[i] for i in valid_indices]
        result_metadata = [self.metadata[i] for i in valid_indices]
        result_ids = [self.document_ids[i] for i in valid_indices]
        
        return {
            'documents': result_docs,
            'metadatas': result_metadata,
            'ids': result_ids
        }
    
    def save(self):
        """Save the vector store to disk."""
        try:
            data = {
                'documents': self.documents,
                'embeddings': [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in self.embeddings],
                'metadata': self.metadata,
                'document_ids': self.document_ids,
                'collection_name': self.collection_name,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Warning: Could not save vector store: {e}")
    
    def load(self):
        """Load the vector store from disk."""
        try:
            if os.path.exists(self.persist_file):
                with open(self.persist_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.documents = data.get('documents', [])
                self.embeddings = [np.array(emb) for emb in data.get('embeddings', [])]
                self.metadata = data.get('metadata', [])
                self.document_ids = data.get('document_ids', [])
                
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            # Initialize empty store
            self.documents = []
            self.embeddings = []
            self.metadata = []
            self.document_ids = []
    
    def clear(self):
        """Clear all data from the store."""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.document_ids = []
        self.save()


class SimpleVectorDB:
    """Simple vector database wrapper that mimics ChromaDB interface."""
    
    def __init__(self, persist_directory: str = "data/vector_store"):
        self.persist_directory = persist_directory
        self.collections = {}
    
    def get_or_create_collection(self, name: str) -> SimpleVectorStore:
        """Get or create a collection."""
        if name not in self.collections:
            self.collections[name] = SimpleVectorStore(name, self.persist_directory)
        return self.collections[name]
    
    def delete_collection(self, name: str):
        """Delete a collection."""
        if name in self.collections:
            del self.collections[name]
            # Also delete the file
            persist_file = os.path.join(self.persist_directory, f"{name}.json")
            if os.path.exists(persist_file):
                os.remove(persist_file)


# Create a simple client that mimics ChromaDB's PersistentClient
class SimpleClient:
    """Simple client that mimics ChromaDB's PersistentClient."""
    
    def __init__(self, path: str = "data/vector_store"):
        self.path = path
        self.db = SimpleVectorDB(path)
    
    def get_or_create_collection(self, name: str, **kwargs) -> SimpleVectorStore:
        """Get or create a collection."""
        return self.db.get_or_create_collection(name)
    
    def delete_collection(self, name: str):
        """Delete a collection."""
        self.db.delete_collection(name)


# Function to create a simple client (mimics chromadb.PersistentClient)
def PersistentClient(path: str = "data/vector_store") -> SimpleClient:
    """Create a persistent client."""
    return SimpleClient(path)


# For backward compatibility, make the module work like chromadb
def Client(**kwargs) -> SimpleClient:
    """Create a client."""
    return SimpleClient(kwargs.get('path', 'data/vector_store'))
