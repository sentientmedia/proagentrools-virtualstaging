#!/usr/bin/env python3
"""
Focused test for AI Content Generation with OpenAI Fallback
Testing the three endpoints mentioned in the review request:
1. Generate Module Content - POST /api/listings/{listing_id}/modules/{module_name}/generate
2. Chat Improve Module - POST /api/listings/{listing_id}/modules/{module_name}/chat
3. Interior Design Processing - POST /api/listings/{listing_id}/interior-design/process
"""

import requests
import json
import uuid
from datetime import datetime
from PIL import Image
import io

class AIContentTester:
    def __init__(self, base_url="https://proagent-realty.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.user_token = None
        self.test_listing_id = None
        self.test_image_ids = []

    def create_test_image(self):
        """Create a simple test image for upload testing"""
        img = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        request_headers = {}
        if data and not files:
            request_headers['Content-Type'] = 'application/json'
        
        if headers:
            request_headers.update(headers)

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=request_headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, headers=headers or {})
                else:
                    response = requests.post(url, json=data, headers=request_headers)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=request_headers)
            elif method == 'DELETE':
                response = requests.delete(url, headers=request_headers)

            success = response.status_code == expected_status
            if success:
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:300]}...")
                except:
                    print(f"   Response: {response.text[:300]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:500]}...")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def setup_test_user(self):
        """Create a test user and get authentication token"""
        test_email = f"ai_test_{uuid.uuid4().hex[:8]}@example.com"
        test_data = {
            "email": test_email,
            "password": "testpassword123",
            "full_name": "AI Test User"
        }
        
        success, response = self.run_test(
            "User Registration for AI Tests",
            "POST",
            "auth/register",
            200,
            data=test_data
        )
        
        if success and response:
            self.user_token = response['access_token']
            print(f"✅ Test user created with token")
            return True
        
        return False

    def create_test_listing(self):
        """Create a test listing for AI content generation"""
        if not self.user_token:
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        test_data = {
            "property_details": {
                "address": "123 AI Test Street",
                "city": "San Francisco", 
                "state": "CA",
                "zip_code": "94102",
                "beds": 3,
                "baths": 2.5,
                "sqft": 2000,
                "property_type": "Single Family",
                "listing_price": 1200000
            },
            "description": "Beautiful test property for AI content generation",
            "selected_tool_ids": ["listing_luxe_gpt", "social_snippets_studio"],
            "agent_notes": "Test listing for AI content generation"
        }
        
        success, response = self.run_test(
            "Create Test Listing",
            "POST",
            "listings",
            200,
            data=test_data,
            headers=headers
        )
        
        if success and response:
            self.test_listing_id = response['id']
            print(f"✅ Test listing created: {self.test_listing_id}")
            return True
        
        return False

    def upload_test_images(self):
        """Upload test images to the listing"""
        if not self.user_token or not self.test_listing_id:
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Create multiple test images
        test_image1 = self.create_test_image()
        test_image2 = self.create_test_image()
        
        files = [
            ('files', ('test_image1.jpg', test_image1, 'image/jpeg')),
            ('files', ('test_image2.jpg', test_image2, 'image/jpeg'))
        ]
        
        success, response = self.run_test(
            "Upload Test Images",
            "POST",
            f"listings/{self.test_listing_id}/images/upload",
            200,
            files=files,
            headers=headers
        )
        
        if success and response:
            uploaded_images = response.get('images', [])
            self.test_image_ids = [img['id'] for img in uploaded_images]
            print(f"✅ Uploaded {len(self.test_image_ids)} test images")
            return True
        
        return False

    def test_generate_module_content(self):
        """Test POST /api/listings/{listing_id}/modules/{module_name}/generate with OpenAI fallback"""
        if not self.user_token or not self.test_listing_id:
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Test generating content for listing_copy module
        test_data = {
            "module_name": "listing_copy",
            "additional_context": "This is a luxury property with modern amenities and mountain views"
        }
        
        success, response = self.run_test(
            "Generate Module Content (listing_copy) - OpenAI Fallback",
            "POST",
            f"listings/{self.test_listing_id}/modules/listing_copy/generate",
            200,
            data=test_data,
            headers=headers
        )
        
        if success and response:
            # Verify response structure
            required_fields = ['content', 'credits_used', 'remaining_credits']
            for field in required_fields:
                if field not in response:
                    print(f"❌ Missing required field '{field}' in response")
                    return False
            
            # Verify credits were deducted (should be 1 credit)
            if response.get('credits_used') != 1:
                print(f"❌ Expected 1 credit used, got {response.get('credits_used')}")
                return False
            
            # Verify content was generated
            content = response.get('content', '')
            if not content or len(content) < 50:
                print(f"❌ Generated content too short or empty: {len(content)} chars")
                return False
            
            print(f"✅ Module content generated successfully")
            print(f"   Credits used: {response.get('credits_used')}")
            print(f"   Remaining credits: {response.get('remaining_credits')}")
            print(f"   Content length: {len(content)} characters")
            return True
        
        return success

    def test_chat_improve_module(self):
        """Test POST /api/listings/{listing_id}/modules/{module_name}/chat with OpenAI fallback"""
        if not self.user_token or not self.test_listing_id:
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # First generate some content for marketing_copy module
        generate_data = {
            "module_name": "marketing_copy",
            "additional_context": "Focus on the property's investment potential and luxury features"
        }
        
        generate_success, generate_response = self.run_test(
            "Generate Marketing Copy for Chat Test",
            "POST",
            f"listings/{self.test_listing_id}/modules/marketing_copy/generate",
            200,
            data=generate_data,
            headers=headers
        )
        
        if not generate_success:
            print("❌ Could not generate initial content for chat test")
            return False
        
        # Now test chat improvement
        chat_data = {
            "message": "Make it more exciting and add emphasis on luxury features and location benefits",
            "module_name": "marketing_copy"
        }
        
        success, response = self.run_test(
            "Chat Improve Module (marketing_copy) - OpenAI Fallback",
            "POST",
            f"listings/{self.test_listing_id}/modules/marketing_copy/chat",
            200,
            data=chat_data,
            headers=headers
        )
        
        if success and response:
            # Verify response structure
            required_fields = ['suggestion', 'credits_used', 'remaining_credits', 'chat_history']
            for field in required_fields:
                if field not in response:
                    print(f"❌ Missing required field '{field}' in response")
                    return False
            
            # Verify credits were deducted (should be 1 credit)
            if response.get('credits_used') != 1:
                print(f"❌ Expected 1 credit used, got {response.get('credits_used')}")
                return False
            
            # Verify AI suggestion was generated
            suggestion = response.get('suggestion', '')
            if not suggestion or len(suggestion) < 20:
                print(f"❌ AI suggestion too short or empty: {len(suggestion)} chars")
                return False
            
            # Verify chat history was stored
            chat_history = response.get('chat_history', [])
            if len(chat_history) < 2:  # Should have user message and AI response
                print(f"❌ Chat history incomplete: {len(chat_history)} messages")
                return False
            
            print(f"✅ Chat improvement successful")
            print(f"   Credits used: {response.get('credits_used')}")
            print(f"   Remaining credits: {response.get('remaining_credits')}")
            print(f"   Suggestion length: {len(suggestion)} characters")
            print(f"   Chat history: {len(chat_history)} messages")
            return True
        
        return success

    def test_interior_design_processing(self):
        """Test POST /api/listings/{listing_id}/interior-design/process with uploaded images"""
        if not self.user_token or not self.test_listing_id or not self.test_image_ids:
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Test interior design processing with uploaded images
        process_data = {
            "image_ids": self.test_image_ids[:2],  # Use first 2 images
            "room_type": "living_room",
            "designer": "alessia_duval",
            "color_scheme": "glacial_muse"
        }
        
        success, response = self.run_test(
            "Process Listing Interior Design",
            "POST",
            f"listings/{self.test_listing_id}/interior-design/process",
            200,
            data=process_data,
            headers=headers
        )
        
        if success and response:
            # Verify response structure
            required_fields = ['success', 'credits_used', 'remaining_credits', 'processed_images']
            for field in required_fields:
                if field not in response:
                    print(f"❌ Missing required field '{field}' in response")
                    return False
            
            # Verify credits were calculated correctly (5 per image)
            expected_credits = len(process_data['image_ids']) * 5
            if response.get('credits_used') != expected_credits:
                print(f"❌ Expected {expected_credits} credits used, got {response.get('credits_used')}")
                return False
            
            # Verify processing was initiated
            processed_images = response.get('processed_images', 0)
            if processed_images != len(process_data['image_ids']):
                print(f"❌ Expected {len(process_data['image_ids'])} processed images, got {processed_images}")
                return False
            
            print(f"✅ Interior design processing initiated successfully")
            print(f"   Credits used: {response.get('credits_used')}")
            print(f"   Remaining credits: {response.get('remaining_credits')}")
            print(f"   Images processed: {processed_images}")
            return True
        
        return success

    def test_authentication_requirements(self):
        """Test that all endpoints require authentication"""
        print("\n🔐 Testing Authentication Requirements...")
        
        endpoints_to_test = [
            ("POST", f"listings/fake-id/modules/listing_copy/generate", "Generate Module Content"),
            ("POST", f"listings/fake-id/modules/marketing_copy/chat", "Chat Improve Module"),
            ("POST", f"listings/fake-id/interior-design/process", "Interior Design Processing")
        ]
        
        all_protected = True
        
        for method, endpoint, name in endpoints_to_test:
            success, response = self.run_test(
                f"{name} Without Auth (Should Fail)",
                method,
                endpoint,
                401,
                data={"test": "data"}
            )
            
            if not success:
                print(f"❌ {name} endpoint not properly protected")
                all_protected = False
            else:
                print(f"✅ {name} endpoint properly requires authentication")
        
        return all_protected

    def run_all_tests(self):
        """Run all AI content generation tests"""
        print("🚀 Starting AI Content Generation Tests with OpenAI Fallback")
        print("="*80)
        
        tests_passed = 0
        tests_total = 0
        
        # Setup
        print("\n📋 SETUP PHASE")
        if not self.setup_test_user():
            print("❌ Failed to setup test user")
            return False
        
        if not self.create_test_listing():
            print("❌ Failed to create test listing")
            return False
        
        if not self.upload_test_images():
            print("❌ Failed to upload test images")
            return False
        
        # Main tests
        print("\n🤖 AI CONTENT GENERATION TESTS")
        
        test_functions = [
            ("Generate Module Content with OpenAI Fallback", self.test_generate_module_content),
            ("Chat Improve Module with OpenAI Fallback", self.test_chat_improve_module),
            ("Interior Design Processing with Images", self.test_interior_design_processing),
            ("Authentication Requirements", self.test_authentication_requirements)
        ]
        
        for test_name, test_func in test_functions:
            tests_total += 1
            print(f"\n🔍 Running: {test_name}")
            try:
                if test_func():
                    tests_passed += 1
                    print(f"✅ {test_name} - PASSED")
                else:
                    print(f"❌ {test_name} - FAILED")
            except Exception as e:
                print(f"❌ {test_name} - ERROR: {str(e)}")
        
        # Summary
        print(f"\n" + "="*80)
        print(f"📊 AI CONTENT GENERATION TESTS SUMMARY")
        print(f"   Passed: {tests_passed}/{tests_total} ({tests_passed/tests_total*100:.1f}%)")
        print("="*80)
        
        if tests_passed == tests_total:
            print("🎉 All AI content generation tests passed!")
            print("✅ OpenAI API key fallback is working correctly")
            return True
        else:
            print(f"⚠️ {tests_total - tests_passed} tests failed")
            return False

if __name__ == "__main__":
    tester = AIContentTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)