#!/usr/bin/env python3
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
