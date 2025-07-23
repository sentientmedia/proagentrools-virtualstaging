import requests
import sys
import json
import io
from datetime import datetime
from PIL import Image
import tempfile
import os

class ProAgentToolsAPITester:
    def __init__(self, base_url="https://7f9f2de6-2fb0-4a74-af1b-8d15dbd9c892.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {}
        if data and not files:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_image(self):
        """Create a simple test image for upload testing"""
        # Create a simple 100x100 RGB image
        img = Image.new('RGB', (100, 100), color='red')
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        return img_buffer

    def test_health_check(self):
        """Test health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "health",
            200
        )
        return success

    def test_root_endpoint(self):
        """Test root API endpoint"""
        success, response = self.run_test(
            "Root Endpoint",
            "GET",
            "",
            200
        )
        return success

    def test_available_concepts(self):
        """Test available GPT concepts endpoint"""
        success, response = self.run_test(
            "Available GPT Concepts",
            "GET",
            "gpt-concepts/available",
            200
        )
        return success

    def test_property_description(self):
        """Test property description generation"""
        test_data = {
            "property_details": "3 bedroom, 2 bathroom house in downtown Seattle with modern kitchen, hardwood floors, and mountain views"
        }
        
        success, response = self.run_test(
            "Property Description Generator",
            "POST",
            "gpt-concepts/property-description",
            200,
            data=test_data
        )
        return success

    def test_market_analysis(self):
        """Test market analysis generation"""
        test_data = {
            "location": "Downtown Seattle",
            "property_type": "Condo"
        }
        
        success, response = self.run_test(
            "Market Analysis Generator",
            "POST",
            "gpt-concepts/market-analysis",
            200,
            data=test_data
        )
        return success

    def test_email_template(self):
        """Test email template generation"""
        test_data = {
            "email_type": "Follow-up",
            "context": "Client showed interest in a downtown condo listing"
        }
        
        success, response = self.run_test(
            "Email Template Generator",
            "POST",
            "gpt-concepts/email-template",
            200,
            data=test_data
        )
        return success

    def test_interior_design_upload(self):
        """Test interior design image processing"""
        # Create test image
        test_image = self.create_test_image()
        
        files = {
            'file': ('test_interior.jpg', test_image, 'image/jpeg')
        }
        
        success, response = self.run_test(
            "Interior Design Image Processing",
            "POST",
            "interior-design/process",
            200,
            files=files
        )
        return success

    def test_interior_design_history(self):
        """Test interior design history endpoint"""
        success, response = self.run_test(
            "Interior Design History",
            "GET",
            "interior-design/history",
            200
        )
        return success

    def test_gpt_concepts_history(self):
        """Test GPT concepts history endpoint"""
        success, response = self.run_test(
            "GPT Concepts History",
            "GET",
            "gpt-concepts/history",
            200
        )
        return success

    def test_invalid_endpoints(self):
        """Test invalid endpoints return proper errors"""
        success, response = self.run_test(
            "Invalid Endpoint (404 Test)",
            "GET",
            "nonexistent-endpoint",
            404
        )
        return success

def main():
    print("🚀 Starting ProAgentTools API Testing...")
    print("=" * 60)
    
    # Setup
    tester = ProAgentToolsAPITester()
    
    # Run basic endpoint tests
    print("\n📋 Testing Basic Endpoints...")
    tester.test_health_check()
    tester.test_root_endpoint()
    tester.test_available_concepts()
    
    # Test GPT Concept tools
    print("\n🤖 Testing GPT Concept Tools...")
    tester.test_property_description()
    tester.test_market_analysis()
    tester.test_email_template()
    
    # Test Interior Design functionality
    print("\n🏠 Testing Interior Design Tool...")
    tester.test_interior_design_upload()
    tester.test_interior_design_history()
    
    # Test history endpoints
    print("\n📊 Testing History Endpoints...")
    tester.test_gpt_concepts_history()
    
    # Test error handling
    print("\n❌ Testing Error Handling...")
    tester.test_invalid_endpoints()
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())