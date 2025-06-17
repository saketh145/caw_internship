#!/usr/bin/env python3
"""
Test script for Dynamic Document Processing
Demonstrates real-time document upload, chunking, and embedding storage
"""

import asyncio
import requests
import json
import time
from pathlib import Path

class DynamicProcessingTest:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.token = None
    
    def authenticate(self):
        """Authenticate with the API"""
        print("🔐 Authenticating...")
        response = requests.post(f"{self.api_base_url}/auth/login",
                               json={"username": "admin", "password": "admin123"})
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            print("✅ Authentication successful")
            return True
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            return False
    
    def upload_document(self, file_path: str, background: bool = False):
        """Upload a document for processing"""
        print(f"\n📄 Uploading document: {file_path}")
        print(f"📊 Background processing: {background}")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            data = {"background_processing": background}
            
            start_time = time.time()
            response = requests.post(f"{self.api_base_url}/documents/upload",
                                   files=files, data=data, headers=headers)
            upload_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful in {upload_time:.2f}s")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Status: {result['status']}")
            print(f"   Chunks: {result['chunks_created']}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            if result.get('message'):
                print(f"   Message: {result['message']}")
            return result['document_id']
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    
    def check_document_status(self, document_id: str):
        """Check the processing status of a document"""
        print(f"\n🔍 Checking status for document: {document_id}")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.api_base_url}/documents/{document_id}/status",
                              headers=headers)
        
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Status: {status['status']}")
            print(f"📁 Filename: {status['filename']}")
            print(f"📅 Upload time: {status['upload_time']}")
            
            if status['processing_started']:
                print(f"⏱️ Processing started: {status['processing_started']}")
            
            if status['processing_completed']:
                print(f"✅ Processing completed: {status['processing_completed']}")
            
            if status.get('processing_result'):
                result = status['processing_result']
                print(f"📄 Chunks created: {result['chunks_created']}")
                print(f"🧠 Embeddings: {result['embeddings_generated']}")
                print(f"⏱️ Processing time: {result['processing_time']}s")
            
            return status
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return None
    
    def list_documents(self):
        """List all documents in the system"""
        print(f"\n📋 Listing all documents...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.api_base_url}/documents", headers=headers)
        
        if response.status_code == 200:
            documents = response.json()
            print(f"📊 Found {len(documents)} documents:")
            
            for doc in documents:
                print(f"   📄 {doc['filename']} ({doc['status']}) - {doc['file_size']} bytes")
            
            return documents
        else:
            print(f"❌ Document listing failed: {response.status_code}")
            return []
    
    def get_processing_stats(self):
        """Get processing statistics"""
        print(f"\n📈 Getting processing statistics...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.api_base_url}/documents/processing/stats",
                              headers=headers)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Processing Statistics:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Completed: {stats['completed']}")
            print(f"   Processing: {stats['processing']}")
            print(f"   Queued: {stats['queued']}")
            print(f"   Failed: {stats['failed']}")
            print(f"   Queue size: {stats['queue_size']}")
            print(f"   Background processor: {'Running' if stats['background_processor_running'] else 'Stopped'}")
            
            return stats
        else:
            print(f"❌ Stats retrieval failed: {response.status_code}")
            return None
    
    def test_qa_with_document(self, question: str):
        """Test Q&A with uploaded documents"""
        print(f"\n❓ Testing Q&A: {question}")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.post(f"{self.api_base_url}/qa/ask",
                               json={"question": question, "session_id": "dynamic_test"},
                               headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Q&A Response:")
            print(f"   Confidence: {result['confidence_score']:.2f}")
            print(f"   Sources: {len(result['sources'])}")
            print(f"   Answer: {result['answer'][:200]}...")
            
            if result['sources']:
                print(f"   Source files: {[s.get('metadata', {}).get('source_file', 'unknown') for s in result['sources']]}")
            
            return result
        else:
            print(f"❌ Q&A failed: {response.status_code}")
            return None

def main():
    """Main test function"""
    print("🧪 DYNAMIC DOCUMENT PROCESSING TEST")
    print("=" * 50)
    
    test = DynamicProcessingTest()
    
    # Step 1: Authenticate
    if not test.authenticate():
        return
    
    # Step 2: Get initial stats
    test.get_processing_stats()
    
    # Step 3: Upload documents (both foreground and background)
    test_files = [
        "test_documents/sample_document.txt",
        "test_documents/machine_learning_guide.txt"
    ]
    
    uploaded_docs = []
    
    for file_path in test_files:
        if Path(file_path).exists():
            # Upload one in foreground, one in background
            background = len(uploaded_docs) % 2 == 1
            doc_id = test.upload_document(file_path, background=background)
            if doc_id:
                uploaded_docs.append(doc_id)
        else:
            print(f"⚠️ Test file not found: {file_path}")
    
    # Step 4: Check status of uploaded documents
    for doc_id in uploaded_docs:
        test.check_document_status(doc_id)
    
    # Step 5: Wait for background processing if any
    if uploaded_docs:
        print(f"\n⏳ Waiting for background processing...")
        time.sleep(5)
        
        # Check stats again
        test.get_processing_stats()
    
    # Step 6: List all documents
    test.list_documents()
    
    # Step 7: Test Q&A with new documents
    test.test_qa_with_document("What is machine learning?")
    
    print(f"\n✅ Dynamic processing test completed!")

if __name__ == "__main__":
    main()
