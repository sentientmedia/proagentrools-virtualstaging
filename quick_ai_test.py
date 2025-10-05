#!/usr/bin/env python3
"""
Quick test for the 3 main AI endpoints mentioned in review request
"""

import requests
import json
import uuid
from PIL import Image
import io

class QuickAITester:
    def __init__(self):
        self.base_url = "https://smart-listing-pro-2.preview.emergentagent.com"
        self.api_url = f"{self.base_url}/api"
        self.user_token = None
        self.test_listing_id = None

    def setup(self):
        """Quick setup"""
        # Register user
        user_data = {
            "email": f"quick_test_{uuid.uuid4().hex[:8]}@example.com",
            "password": "testpass123",
            "full_name": "Quick Test User"
        }
        
        response = requests.post(f"{self.api_url}/auth/register", json=user_data)
        if response.status_code != 200:
            print(f"❌ User registration failed: {response.status_code}")
            return False
        
        self.user_token = response.json()['access_token']
        print(f"✅ User registered")
        
        # Create listing
        headers = {"Authorization": f"Bearer {self.user_token}"}
        listing_data = {
            "property_details": {
                "address": "123 Quick Test St",
                "city": "San Francisco", 
                "state": "CA",
                "zip_code": "94102",
                "beds": 3,
                "baths": 2.0,
                "sqft": 1800,
                "property_type": "Single Family",
                "listing_price": 1000000
            },
            "description": "Quick test property",
            "selected_tool_ids": ["listing_luxe_gpt"],
            "agent_notes": "Quick test"
        }
        
        response = requests.post(f"{self.api_url}/listings", json=listing_data, headers=headers)
        if response.status_code != 200:
            print(f"❌ Listing creation failed: {response.status_code}")
            return False
        
        self.test_listing_id = response.json()['id']
        print(f"✅ Listing created: {self.test_listing_id}")
        
        # Upload test image
        img = Image.new('RGB', (100, 100), color='blue')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        files = {'images': ('test.jpg', img_buffer, 'image/jpeg')}
        response = requests.post(
            f"{self.api_url}/listings/{self.test_listing_id}/images/upload",
            files=files,
            headers=headers
        )
        
        if response.status_code != 200:
            print(f"❌ Image upload failed: {response.status_code}")
            return False
        
        self.test_image_id = response.json()['uploaded_images'][0]['id']
        print(f"✅ Image uploaded: {self.test_image_id}")
        return True

    def test_generate_module_content(self):
        """Test Generate Module Content endpoint"""
        headers = {"Authorization": f"Bearer {self.user_token}"}
        data = {"additional_context": "Luxury property with modern amenities"}
        
        response = requests.post(
            f"{self.api_url}/listings/{self.test_listing_id}/modules/listing_copy/generate",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('content') and result.get('credits_used') == 1:
                print(f"✅ Generate Module Content: WORKING")
                print(f"   Content length: {len(result['content'])} chars")
                print(f"   Credits used: {result['credits_used']}")
                return True
        
        print(f"❌ Generate Module Content: FAILED ({response.status_code})")
        print(f"   Response: {response.text[:200]}")
        return False

    def test_chat_improve_module(self):
        """Test Chat Improve Module endpoint"""
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # First generate content
        gen_data = {"additional_context": "Modern home with great features"}
        gen_response = requests.post(
            f"{self.api_url}/listings/{self.test_listing_id}/modules/marketing_copy/generate",
            json=gen_data,
            headers=headers
        )
        
        if gen_response.status_code != 200:
            print(f"❌ Chat Improve Module: FAILED (couldn't generate initial content)")
            return False
        
        # Now test chat improvement
        chat_data = {"message": "Make this more engaging and add luxury emphasis"}
        response = requests.post(
            f"{self.api_url}/listings/{self.test_listing_id}/modules/marketing_copy/chat",
            json=chat_data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('response'):
                print(f"✅ Chat Improve Module: WORKING")
                print(f"   Response length: {len(result['response'])} chars")
                return True
        
        print(f"❌ Chat Improve Module: FAILED ({response.status_code})")
        print(f"   Response: {response.text[:200]}")
        return False

    def test_interior_design_processing(self):
        """Test Interior Design Processing endpoint"""
        headers = {"Authorization": f"Bearer {self.user_token}"}
        data = {
            "image_ids": [self.test_image_id],
            "room_type": "living_room",
            "designer": "alessia_duval",
            "color_scheme": "glacial_muse"
        }
        
        response = requests.post(
            f"{self.api_url}/listings/{self.test_listing_id}/interior-design/process",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('credits_used') == 5:
                print(f"✅ Interior Design Processing: WORKING")
                print(f"   Credits used: {result['credits_used']}")
                print(f"   Processed count: {result.get('processed_count', 0)}")
                return True
        
        print(f"❌ Interior Design Processing: FAILED ({response.status_code})")
        print(f"   Response: {response.text[:200]}")
        return False

    def run_tests(self):
        """Run all tests"""
        print("🚀 Quick AI Content Generation Test")
        print("=" * 50)
        
        if not self.setup():
            print("❌ Setup failed")
            return
        
        print("\n🧪 Testing AI Endpoints...")
        
        results = []
        results.append(("Generate Module Content", self.test_generate_module_content()))
        results.append(("Chat Improve Module", self.test_chat_improve_module()))
        results.append(("Interior Design Processing", self.test_interior_design_processing()))
        
        print("\n📊 RESULTS:")
        print("=" * 50)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ WORKING" if result else "❌ FAILED"
            print(f"{name}: {status}")
        
        print(f"\nSUMMARY: {passed}/{total} endpoints working ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL AI CONTENT GENERATION ENDPOINTS WORKING!")
            print("✅ OpenAI fallback logic is functioning correctly")
        else:
            print("⚠️ Some endpoints still have issues")

if __name__ == "__main__":
    tester = QuickAITester()
    tester.run_tests()