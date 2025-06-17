#!/usr/bin/env python3
"""
Production Finalizer for Intelligent Document Q&A System
========================================================

This script performs final production tasks:
1. Creates production configuration
2. Generates system analytics
3. Creates monitoring setup
4. Validates system readiness
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
            print("❌ API is not running. Please start the FastAPI server first.")
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

def create_production_config():
    """Create production configuration files"""
    
    # Environment configuration
    prod_env = {
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "LOG_LEVEL": "INFO",
        "CHROMA_PERSIST_DIRECTORY": "./data/chroma_db",
        "UPLOAD_DIRECTORY": "./uploads",
        "MAX_CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "200",
        "MAX_FILE_SIZE": "50MB",
        "ALLOWED_EXTENSIONS": "pdf,docx,txt,md",
        "CORS_ORIGINS": "http://localhost:3000,http://localhost:8501",
        "API_VERSION": "v1"
    }
    
    # Save production environment
    with open(".env.production", "w") as f:
        for key, value in prod_env.items():
            f.write(f"{key}={value}\n")
    
    print("✅ Created .env.production")
    
    # Docker configuration
    dockerfile_content = """# Production Dockerfile for Document Q&A System
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
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
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile")
    
    # Docker Compose for production
    docker_compose = """version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
    restart: unless-stopped

volumes:
  data:
  uploads:
  logs:
"""
    
    with open("docker-compose.prod.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Created docker-compose.prod.yml")

def create_monitoring_config():
    """Create monitoring and logging configuration"""
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "level": "INFO",
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/app.log",
                "maxBytes": 10485760,
                "backupCount": 5
            },
            "error_file": {
                "level": "ERROR",
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/error.log",
                "maxBytes": 10485760,
                "backupCount": 5
            }
        },
        "loggers": {
            "": {
                "handlers": ["default", "file", "error_file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    os.makedirs("logs", exist_ok=True)
    with open("logging_config.json", "w") as f:
        json.dump(logging_config, f, indent=2)
    
    print("✅ Created logging configuration")
    
    # Health check script
    health_check = """#!/usr/bin/env python3
import requests
import sys
import time

def health_check():
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    if health_check():
        sys.exit(0)
    else:
        sys.exit(1)
"""
      with open("health_check.py", "w", encoding="utf-8") as f:
        f.write(health_check)
    
    print("✅ Created health check script")

def generate_system_analytics():
    """Generate comprehensive system analytics"""
    
    print("\n📊 Generating System Analytics...")
    
    stats = get_system_stats()
    if not stats:
        print("❌ Could not generate analytics - API not available")
        return
    
    analytics = {
        "generated_at": datetime.now().isoformat(),
        "system_status": "operational",
        "document_processing": stats["processing_stats"],
        "document_summary": {
            "total_documents": stats["document_count"],
            "document_types": {},
            "processing_status": {}
        },
        "performance_metrics": {
            "avg_processing_time": 0,
            "total_chunks_processed": 0,
            "storage_usage": "N/A"
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
    
    print("✅ Generated system analytics")
    
    # Print summary
    print("\n📈 System Analytics Summary:")
    print(f"   Total Documents: {analytics['document_summary']['total_documents']}")
    print(f"   Document Types: {analytics['document_summary']['document_types']}")
    print(f"   Processing Status: {analytics['document_summary']['processing_status']}")

def create_deployment_guide():
    """Create deployment guide"""
    
    guide = """# Production Deployment Guide

## Quick Start

1. **Environment Setup**:
   ```bash
   cp .env.production .env
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Services**:
   ```bash
   # API Server
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000

   # Frontend (separate terminal)
   streamlit run src/frontend/streamlit_app.py --server.port 8501
   ```

## Docker Deployment

1. **Build and Run**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Check Status**:
   ```bash
   docker-compose -f docker-compose.prod.yml ps
   ```

3. **View Logs**:
   ```bash
   docker-compose -f docker-compose.prod.yml logs -f
   ```

## Health Monitoring

- **API Health**: http://localhost:8000/health
- **Frontend**: http://localhost:8501
- **Health Check Script**: `python health_check.py`

## API Endpoints

- **Upload Document**: POST `/api/v1/upload`
- **Ask Question**: POST `/api/v1/ask`
- **List Documents**: GET `/api/v1/documents`
- **Processing Stats**: GET `/api/v1/processing-stats`

## Configuration

- **Environment**: `.env.production`
- **Logging**: `logging_config.json`
- **Docker**: `docker-compose.prod.yml`

## Troubleshooting

1. **API not responding**: Check if port 8000 is available
2. **Upload failures**: Verify `uploads/` directory permissions
3. **Database issues**: Check ChromaDB data directory
4. **Memory issues**: Adjust chunk size and overlap settings

## Security Considerations

- Change default API keys and secrets
- Enable HTTPS in production
- Configure proper CORS origins
- Set up authentication if needed
- Regular security updates

## Performance Optimization

- Monitor memory usage during document processing
- Adjust chunk sizes based on document types
- Configure appropriate timeouts
- Use connection pooling for high traffic
"""
    
    with open("DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(guide)
    
    print("✅ Created deployment guide")

def validate_production_readiness():
    """Validate system is production ready"""
    
    print("\n🔍 Validating Production Readiness...")
    
    checks = []
    
    # Check API availability
    api_available = check_api_status()
    checks.append(("API Server", "✅ Running" if api_available else "❌ Not running"))
    
    # Check required files
    required_files = [
        "src/api/main.py",
        "src/document_processor/dynamic_processor.py",
        "src/qa_engine/qa_engine.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        checks.append((f"File: {file_path}", "✅ Exists" if exists else "❌ Missing"))
    
    # Check directories
    required_dirs = ["data", "uploads", "logs"]
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        checks.append((f"Directory: {dir_path}", "✅ Exists" if exists else "❌ Missing"))
    
    # Check configuration files
    config_files = [".env.production", "Dockerfile", "docker-compose.prod.yml"]
    for config_file in config_files:
        exists = os.path.exists(config_file)
        checks.append((f"Config: {config_file}", "✅ Created" if exists else "❌ Missing"))
    
    # Print results
    print("\n📋 Production Readiness Checklist:")
    for item, status in checks:
        print(f"   {item}: {status}")
    
    # Overall status
    all_critical_passed = api_available and all(os.path.exists(f) for f in required_files[:3])
    
    print(f"\n🏁 Overall Status: {'✅ PRODUCTION READY' if all_critical_passed else '⚠️  NEEDS ATTENTION'}")
    
    return all_critical_passed

def main():
    """Main execution function"""
    
    print("🚀 Final Production Enhancements")
    print("=" * 50)
    
    try:
        # Create production configurations
        print("\n1. Creating Production Configuration...")
        create_production_config()
        
        # Create monitoring setup
        print("\n2. Setting up Monitoring...")
        create_monitoring_config()
        
        # Generate analytics
        print("\n3. Generating Analytics...")
        generate_system_analytics()
        
        # Create deployment guide
        print("\n4. Creating Deployment Guide...")
        create_deployment_guide()
        
        # Validate readiness
        print("\n5. Validating Production Readiness...")
        is_ready = validate_production_readiness()
        
        # Final summary
        print("\n" + "=" * 50)
        print("🎉 Production Enhancement Complete!")
        print("\nWhat's been created:")
        print("   📁 Production configurations (.env.production, Dockerfile)")
        print("   📊 System analytics (system_analytics.json)")
        print("   🔍 Health monitoring (health_check.py)")
        print("   📖 Deployment guide (DEPLOYMENT_GUIDE.md)")
        print("   📋 Production validation results")
        
        if is_ready:
            print("\n✅ System is PRODUCTION READY!")
            print("   You can now deploy using Docker or manual setup.")
        else:
            print("\n⚠️  Please address the issues above before deployment.")
        
        print("\nNext steps:")
        print("   1. Review configuration files")
        print("   2. Test deployment in staging environment")
        print("   3. Push to GitHub repository")
        print("   4. Deploy to production")
        
    except Exception as e:
        print(f"\n❌ Error during production enhancement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
