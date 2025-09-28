# test_api.py - Test the API endpoints
import requests
import time
import json

def test_endpoints():
    base_url = "http://localhost:8000"
    
    # Test basic health endpoint
    try:
        print("Testing /health endpoint...")
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing /health: {e}")
        return
    
    # Test API health endpoint
    try:
        print("Testing /api/health endpoint...")
        response = requests.get(f"{base_url}/api/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing /api/health: {e}")
    
    # Test system status endpoint
    try:
        print("Testing /api/system/status endpoint...")
        response = requests.get(f"{base_url}/api/system/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing /api/system/status: {e}")
    
    # Test agents config endpoint
    try:
        print("Testing /api/agents/config endpoint...")
        response = requests.get(f"{base_url}/api/agents/config")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing /api/agents/config: {e}")
    
    # Test invoice upload endpoint
    try:
        print("Testing /api/invoices/upload endpoint...")
        test_invoice = {
            "invoice_number": "TEST-001",
            "vendor": {"name": "Test Vendor", "address": "123 Test St"},
            "invoice_date": "2024-01-01",
            "total_amount": 1000.00,
            "line_items": [{"description": "Test Item", "quantity": 1, "unit_price": 1000.00, "total": 1000.00}]
        }
        
        response = requests.post(f"{base_url}/api/invoices/upload", 
                               data={"invoice_data": json.dumps(test_invoice)})
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing /api/invoices/upload: {e}")
    
    # Test get invoices endpoint
    try:
        print("Testing /api/invoices endpoint...")
        response = requests.get(f"{base_url}/api/invoices")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing /api/invoices: {e}")

if __name__ == "__main__":
    print("Starting API tests...")
    test_endpoints()
    print("API tests completed!")