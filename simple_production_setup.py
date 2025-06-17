#!/usr/bin/env python3
"""
Simple Production Finalizer for Windows
======================================

Creates production configurations without Unicode characters.
"""

import os
import json
import requests
import time
from datetime import datetime
from pathlib import Path

def check_api_status():
    """Check if the API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        if not check_api_status():
            print("API is not running. Please start the FastAPI server first.")
            return None
            
        # Get processing stats
        stats_response = requests.get("http://localhost:8000/api/v1/processing-stats")
        processing_stats = stats_response.json() if stats_response.status_code == 200 else {}
        
        # Get document list
        docs_response = requests.get("http://localhost:8000/api/v1/documents")
        documents = docs_response.json() if docs_response.status_code == 200 else []
        
        return {
            "processing_stats": processing_stats,
            "document_count": len(documents),
            "documents": documents,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error getting system stats: {e}")
        return None

def create_production_files():
    """Create all production files"""
    
    print("Creating production configuration files...")
    
    # Environment configuration
    prod_env = """ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
UPLOAD_DIRECTORY=./uploads
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=50MB
ALLOWED_EXTENSIONS=pdf,docx,txt,md
CORS_ORIGINS=http://localhost:3000,http://localhost:8501
API_VERSION=v1
"""
    
    with open(".env.production", "w") as f:
        f.write(prod_env)
    
    print("Created .env.production")
    
    # Simple health check script
    health_check = """#!/usr/bin/env python3
import requests
import sys

def health_check():
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("API is healthy")
            return True
        else:
            print(f"API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    if health_check():
        sys.exit(0)
    else:
        sys.exit(1)
"""
    
    with open("health_check.py", "w") as f:
        f.write(health_check)
    
    print("Created health_check.py")
    
    # Dockerfile
    dockerfile = """# Production Dockerfile for Document Q&A System
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY *.py ./

# Create directories
RUN mkdir -p data uploads logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    print("Created Dockerfile")

def generate_analytics():
    """Generate system analytics"""
    
    print("Generating system analytics...")
    
    stats = get_system_stats()
    if not stats:
        print("Could not generate analytics - API not available")
        return
    
    analytics = {
        "generated_at": datetime.now().isoformat(),
        "system_status": "operational",
        "document_processing": stats["processing_stats"],
        "document_summary": {
            "total_documents": stats["document_count"],
            "document_types": {},
            "processing_status": {}
        }
    }
    
    # Analyze documents
    for doc in stats["documents"]:
        doc_type = doc.get("metadata", {}).get("file_type", "unknown")
        status = doc.get("status", "unknown")
        
        analytics["document_summary"]["document_types"][doc_type] = \
            analytics["document_summary"]["document_types"].get(doc_type, 0) + 1
        
        analytics["document_summary"]["processing_status"][status] = \
            analytics["document_summary"]["processing_status"].get(status, 0) + 1
    
    # Save analytics
    with open("system_analytics.json", "w") as f:
        json.dump(analytics, f, indent=2)
    
    print("Generated system_analytics.json")
    
    # Print summary
    print(f"Total Documents: {analytics['document_summary']['total_documents']}")
    print(f"Document Types: {analytics['document_summary']['document_types']}")
    print(f"Processing Status: {analytics['document_summary']['processing_status']}")

def validate_system():
    """Validate system readiness"""
    
    print("Validating production readiness...")
    
    # Check API
    api_running = check_api_status()
    print(f"API Status: {'Running' if api_running else 'Not running'}")
    
    # Check required files
    required_files = [
        "src/api/main.py",
        "src/document_processor/dynamic_processor.py", 
        "src/qa_engine/qa_engine.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
    else:
        print("All required files present")
    
    # Check directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("uploads", exist_ok=True) 
    os.makedirs("logs", exist_ok=True)
    print("Created required directories")
    
    # Overall status
    is_ready = api_running and not missing_files
    print(f"System Status: {'PRODUCTION READY' if is_ready else 'NEEDS ATTENTION'}")
    
    return is_ready

def main():
    """Main function"""
    
    print("Final Production Setup")
    print("=" * 30)
    
    try:
        # Create production files
        create_production_files()
        print()
        
        # Generate analytics
        generate_analytics()
        print()
        
        # Validate system
        is_ready = validate_system()
        print()
        
        print("Production setup complete!")
        print("Files created:")
        print("- .env.production")
        print("- health_check.py") 
        print("- Dockerfile")
        print("- system_analytics.json")
        
        if is_ready:
            print("System is ready for production deployment!")
        else:
            print("Please address issues before deployment.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
