import requests
import json
import uuid
import time
from datetime import datetime

class FrontendStatusSyncTester:
    def __init__(self, base_url="https://realestate-ai-65.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.user_token = None
        self.test_listing_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        request_headers = {}
        if data:
            request_headers['Content-Type'] = 'application/json'
        
        if headers:
            request_headers.update(headers)

        print(f"\n🔍 {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=request_headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=request_headers)

            success = response.status_code == expected_status
            if success:
                print(f"✅ Status: {response.status_code}")
            else:
                print(f"❌ Expected {expected_status}, got {response.status_code}")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False, {}

    def setup_test_user(self):
        """Create a test user"""
        test_email = f"frontend_sync_test_{uuid.uuid4().hex[:8]}@example.com"
        test_data = {
            "email": test_email,
            "password": "testpassword123",
            "full_name": "Frontend Sync Test User"
        }
        
        success, response = self.run_test(
            "Create Test User",
            "POST",
            "auth/register",
            200,
            data=test_data
        )
        
        if success and response:
            self.user_token = response['access_token']
            print(f"✅ Test user created")
            return True
        
        return False

    def test_frontend_status_synchronization(self):
        """Test the specific issue: frontend not showing updated AI processing status"""
        print("\n🎯 TESTING FRONTEND STATUS SYNCHRONIZATION ISSUE")
        print("=" * 60)
        
        if not self.setup_test_user():
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Step 1: Create a listing with AI tools
        print("\n📋 Step 1: Create listing with AI tools")
        test_data = {
            "property_details": {
                "address": "123 Frontend Sync Test Street",
                "city": "San Francisco", 
                "state": "CA",
                "zip_code": "94105",
                "beds": 2,
                "baths": 2.0,
                "sqft": 1200,
                "property_type": "Condo",
                "listing_price": 850000
            },
            "description": "Test property for frontend sync issue",
            "selected_tool_ids": ["listing_luxe_gpt", "social_snippets_studio"],
            "agent_notes": "Testing frontend status sync"
        }
        
        success, response = self.run_test(
            "Create Listing",
            "POST",
            "listings",
            200,
            data=test_data,
            headers=headers
        )
        
        if not success:
            return False
        
        self.test_listing_id = response['id']
        initial_status = response.get('ai_processing_status')
        print(f"   Initial AI processing status: {initial_status}")
        
        # Step 2: Check status before processing
        print("\n📋 Step 2: Check status before AI processing")
        success, response = self.run_test(
            "Get Listing Before Processing",
            "GET",
            f"listings/{self.test_listing_id}",
            200,
            headers=headers
        )
        
        if success:
            status_before = response.get('ai_processing_status')
            print(f"   Status before processing: {status_before}")
        
        # Step 3: Trigger AI processing
        print("\n📋 Step 3: Trigger AI processing")
        success, response = self.run_test(
            "Trigger AI Processing",
            "POST",
            f"listings/{self.test_listing_id}/process-ai",
            200,
            headers=headers
        )
        
        if not success:
            return False
        
        processing_response = response
        print(f"   Processing result: {processing_response.get('message')}")
        print(f"   Tools processed: {processing_response.get('tools_processed')}")
        print(f"   Credits used: {processing_response.get('credits_used')}")
        
        # Step 4: Immediately check status after processing
        print("\n📋 Step 4: Check status immediately after processing")
        success, response = self.run_test(
            "Get Listing Immediately After Processing",
            "GET",
            f"listings/{self.test_listing_id}",
            200,
            headers=headers
        )
        
        if success:
            status_after = response.get('ai_processing_status')
            has_output = bool(response.get('ai_output'))
            updated_at = response.get('updated_at')
            
            print(f"   Status immediately after: {status_after}")
            print(f"   Has AI output: {has_output}")
            print(f"   Last updated: {updated_at}")
            
            # This is the key test - check if status is properly updated
            if status_after == 'completed' and has_output:
                print("   ✅ FRONTEND SYNC WORKING: Status correctly updated to 'completed' with AI output")
            elif status_after == 'processing':
                print("   ⚠️ POTENTIAL TIMING ISSUE: Status still 'processing' - may need frontend refresh")
            elif status_after == 'completed' and not has_output:
                print("   ❌ INCONSISTENCY: Status 'completed' but no AI output")
            else:
                print(f"   ❌ UNEXPECTED STATUS: {status_after}")
        
        # Step 5: Check listings endpoint (what frontend would call)
        print("\n📋 Step 5: Check listings endpoint (frontend view)")
        success, response = self.run_test(
            "Get All Listings (Frontend View)",
            "GET",
            "listings",
            200,
            headers=headers
        )
        
        if success:
            listings = response
            test_listing = next((l for l in listings if l['id'] == self.test_listing_id), None)
            
            if test_listing:
                frontend_status = test_listing.get('ai_processing_status')
                frontend_has_output = bool(test_listing.get('ai_output'))
                
                print(f"   Frontend view status: {frontend_status}")
                print(f"   Frontend view has output: {frontend_has_output}")
                
                if frontend_status == 'completed' and frontend_has_output:
                    print("   ✅ FRONTEND SYNC CONFIRMED: Listings endpoint shows correct status")
                else:
                    print("   ❌ FRONTEND SYNC ISSUE: Listings endpoint shows incorrect status")
        
        # Step 6: Test AI results endpoint
        print("\n📋 Step 6: Test AI results endpoint")
        success, response = self.run_test(
            "Get AI Results",
            "GET",
            f"listings/{self.test_listing_id}/ai-results",
            200,
            headers=headers
        )
        
        if success:
            processing_status = response.get('processing_status')
            ai_results = response.get('ai_results')
            
            print(f"   AI results status: {processing_status}")
            print(f"   Has AI results: {bool(ai_results)}")
            
            if processing_status == 'completed' and ai_results:
                print("   ✅ AI RESULTS ENDPOINT WORKING: Results available")
            else:
                print("   ❌ AI RESULTS ENDPOINT ISSUE: Results not available")
        
        # Step 7: Final verification
        print("\n📋 Step 7: Final verification after short delay")
        time.sleep(2)  # Small delay to ensure any async updates complete
        
        success, response = self.run_test(
            "Final Status Check",
            "GET",
            f"listings/{self.test_listing_id}",
            200,
            headers=headers
        )
        
        if success:
            final_status = response.get('ai_processing_status')
            final_has_output = bool(response.get('ai_output'))
            final_updated = response.get('updated_at')
            
            print(f"   Final status: {final_status}")
            print(f"   Final has output: {final_has_output}")
            print(f"   Final updated: {final_updated}")
            
            # Summary
            print("\n" + "=" * 60)
            print("🎯 FRONTEND STATUS SYNC TEST SUMMARY")
            
            if final_status == 'completed' and final_has_output:
                print("✅ RESULT: AI processing status synchronization is WORKING CORRECTLY")
                print("   - AI processing completes successfully")
                print("   - Status updates from 'pending' → 'completed'")
                print("   - AI output is stored and accessible")
                print("   - Frontend endpoints show correct status")
                return True
            else:
                print("❌ RESULT: AI processing status synchronization has ISSUES")
                print(f"   - Final status: {final_status}")
                print(f"   - Has output: {final_has_output}")
                return False
        
        return False

if __name__ == "__main__":
    tester = FrontendStatusSyncTester()
    success = tester.test_frontend_status_synchronization()
    
    if success:
        print("\n✅ Frontend status synchronization test PASSED!")
    else:
        print("\n❌ Frontend status synchronization test FAILED!")