#!/usr/bin/env python3

import requests
import json
import uuid

class QuickDeleteTest:
    def __init__(self):
        self.base_url = "https://aihomedesign.preview.emergentagent.com"
        self.api_url = f"{self.base_url}/api"
        self.user_token = None

    def register_user(self):
        """Register a test user"""
        test_email = f"testuser_{uuid.uuid4().hex[:8]}@example.com"
        test_data = {
            "email": test_email,
            "password": "testpassword123",
            "full_name": "Test User"
        }
        
        response = requests.post(f"{self.api_url}/auth/register", json=test_data)
        if response.status_code == 200:
            data = response.json()
            self.user_token = data['access_token']
            print(f"✅ User registered: {test_email}")
            return True
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.text}")
            return False

    def create_listing(self):
        """Create a test listing"""
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        test_data = {
            "property_details": {
                "address": "456 Delete Street",
                "city": "Test City", 
                "state": "CA",
                "zip_code": "90210",
                "beds": 2,
                "baths": 1.0,
                "sqft": 1000,
                "property_type": "Condo",
                "listing_price": 800000
            },
            "description": "Listing to be deleted",
            "selected_tool_ids": ["listing_luxe_gpt"],
            "agent_notes": "Test deletion"
        }
        
        response = requests.post(f"{self.api_url}/listings", json=test_data, headers=headers)
        if response.status_code == 200:
            data = response.json()
            listing_id = data['id']
            print(f"✅ Listing created: {listing_id}")
            return listing_id
        else:
            print(f"❌ Listing creation failed: {response.status_code} - {response.text}")
            return None

    def delete_listing(self, listing_id):
        """Delete the listing"""
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        response = requests.delete(f"{self.api_url}/listings/{listing_id}", headers=headers)
        print(f"DELETE response status: {response.status_code}")
        print(f"DELETE response text: {response.text}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Delete response: {data}")
                return data.get('success', False)
            except:
                print("❌ Could not parse JSON response")
                return False
        else:
            print(f"❌ Delete failed: {response.status_code}")
            return False

    def verify_deletion(self, listing_id):
        """Verify listing was deleted"""
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        response = requests.get(f"{self.api_url}/listings/{listing_id}", headers=headers)
        if response.status_code == 404:
            print("✅ Listing successfully deleted (404 confirmed)")
            return True
        else:
            print(f"❌ Listing still exists: {response.status_code}")
            return False

    def run_test(self):
        """Run the complete delete test"""
        print("🧪 Testing DELETE /api/listings/{id} endpoint...")
        
        # Step 1: Register user
        if not self.register_user():
            return False
        
        # Step 2: Create listing
        listing_id = self.create_listing()
        if not listing_id:
            return False
        
        # Step 3: Delete listing
        if not self.delete_listing(listing_id):
            return False
        
        # Step 4: Verify deletion
        if not self.verify_deletion(listing_id):
            return False
        
        print("🎉 DELETE endpoint test completed successfully!")
        return True

if __name__ == "__main__":
    tester = QuickDeleteTest()
    tester.run_test()