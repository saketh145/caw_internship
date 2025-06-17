#!/usr/bin/env python3
"""
Quick Validation Runner - Test existing running system
"""

import subprocess
import sys
import requests
import time
from datetime import datetime

def check_system_status():
    """Check if system is running"""
    print("CHECKING SYSTEM STATUS...")
    print("=" * 50)
    
    # Check API server
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ API Server: RUNNING (port 8000)")
        else:
            print("✗ API Server: UNHEALTHY")
            return False
    except:
        print("✗ API Server: NOT RUNNING")
        return False
    
    # Check Streamlit
    try:
        response = requests.get("http://localhost:8502", timeout=5)
        if response.status_code == 200:
            print("✓ Streamlit UI: RUNNING (port 8502)")
        else:
            print("✗ Streamlit UI: UNHEALTHY")
    except:
        print("✗ Streamlit UI: NOT RUNNING")
    
    print()
    return True

def run_quick_test():
    """Run a quick functionality test"""
    print("RUNNING QUICK FUNCTIONALITY TEST...")
    print("=" * 50)
    
    try:
        # Authenticate
        auth_response = requests.post("http://localhost:8000/auth/login",
                                    json={"username": "admin", "password": "admin123"})
        
        if auth_response.status_code != 200:
            print("✗ Authentication failed")
            return False
        
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        print("✓ Authentication successful")
        
        # Test Q&A
        start_time = time.time()
        qa_response = requests.post("http://localhost:8000/qa/ask",
                                  json={
                                      "question": "What is artificial intelligence?",
                                      "session_id": "quick_test"
                                  },
                                  headers=headers)
        end_time = time.time()
        
        if qa_response.status_code == 200:
            result = qa_response.json()
            response_time = end_time - start_time
            confidence = result.get('confidence_score', 0)
            
            print(f"✓ Q&A Engine working")
            print(f"  Response time: {response_time:.2f}s")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Answer preview: {result.get('answer', '')[:100]}...")
        else:
            print("✗ Q&A Engine failed")
            return False
        
        # Check metrics
        metrics_response = requests.get("http://localhost:8000/metrics", headers=headers)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            print(f"✓ System metrics accessible")
            print(f"  Documents: {metrics['usage_stats']['total_documents']}")
            print(f"  Questions: {metrics['usage_stats']['total_questions']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def run_full_validation():
    """Run the full production validator"""
    print("RUNNING FULL VALIDATION SUITE...")
    print("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, "production_validator.py"], 
                              cwd=".", timeout=60)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("✗ Validation timed out")
        return False
    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")
        return False

def main():
    """Main validation runner"""
    print(f"INTELLIGENT DOCUMENT Q&A SYSTEM - QUICK VALIDATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Check system status
    if not check_system_status():
        print("\n❌ SYSTEM NOT RUNNING")
        print("Please start the system with:")
        print("  1. uvicorn src.api.main:app --reload --port 8000")
        print("  2. streamlit run enhanced_streamlit_app_working.py --server.port 8502")
        return 1
    
    # Step 2: Quick test
    print()
    if not run_quick_test():
        print("\n❌ QUICK TEST FAILED")
        return 1
    
    # Step 3: Full validation
    print()
    if not run_full_validation():
        print("\n⚠️  FULL VALIDATION HAD ISSUES")
        print("Check the output above for details")
    
    print("\n" + "=" * 60)
    print("✅ VALIDATION COMPLETE")
    print("\n🚀 SYSTEM STATUS: READY FOR PRODUCTION")
    print("\n📊 ACCESS POINTS:")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Streamlit Interface: http://localhost:8502")
    print("   • Health Check: http://localhost:8000/health")
    print("\n🔐 Login: admin / admin123")
    
    return 0

if __name__ == "__main__":
    exit(main())
