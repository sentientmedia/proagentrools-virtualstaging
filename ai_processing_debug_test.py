import requests
import sys
import json
import uuid
import time
from datetime import datetime
import pymongo
from pymongo import MongoClient
import os
from dotenv import load_dotenv

class AIProcessingDebugTester:
    def __init__(self, base_url="https://smart-listing-pro-2.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.user_token = None
        self.test_user_id = None
        self.test_listing_id = None
        
        # Load environment variables for database access
        load_dotenv('/app/backend/.env')
        
        # MongoDB connection for direct database queries
        try:
            mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
            db_name = os.environ.get('DB_NAME', 'proagenttools')
            self.mongo_client = MongoClient(mongo_url)
            self.db = self.mongo_client[db_name]
            print(f"✅ Connected to MongoDB: {db_name}")
        except Exception as e:
            print(f"⚠️ Could not connect to MongoDB: {e}")
            self.mongo_client = None
            self.db = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, headers=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        request_headers = {}
        if data and not files:
            request_headers['Content-Type'] = 'application/json'
        
        # Add authorization headers if provided
        if headers:
            request_headers.update(headers)

        self.tests_run += 1
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
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:300]}...")
                except:
                    print(f"   Response: {response.text[:300]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:300]}...")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def setup_test_user(self):
        """Create a test user for authentication"""
        test_email = f"ai_debug_user_{uuid.uuid4().hex[:8]}@example.com"
        test_data = {
            "email": test_email,
            "password": "testpassword123",
            "full_name": "AI Debug Test User"
        }
        
        success, response = self.run_test(
            "Setup Test User",
            "POST",
            "auth/register",
            200,
            data=test_data
        )
        
        if success and response:
            self.user_token = response['access_token']
            self.test_user_id = response['user']['id']
            print(f"✅ Test user created with ID: {self.test_user_id}")
            return True
        
        return False

    def query_database_listings(self):
        """Query database directly to check recent AI processing"""
        if self.db is None:
            print("⚠️ No database connection available")
            return False
        
        print("\n🔍 DEBUGGING STEP 1: Check Recent AI Processing in Database")
        
        try:
            # Find recent listings with AI processing
            recent_listings = list(self.db.listings.find({
                "created_at": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)}
            }).sort("created_at", -1).limit(10))
            
            print(f"   Found {len(recent_listings)} recent listings")
            
            # Check for listings with different AI processing statuses
            processing_statuses = {}
            completed_with_output = 0
            stuck_in_processing = 0
            
            for listing in recent_listings:
                status = listing.get('ai_processing_status', 'unknown')
                processing_statuses[status] = processing_statuses.get(status, 0) + 1
                
                if status == 'completed' and listing.get('ai_output'):
                    completed_with_output += 1
                    print(f"   ✅ Listing {listing['id'][:8]}... - Status: {status}, Has Output: Yes")
                elif status == 'processing':
                    stuck_in_processing += 1
                    created_time = listing.get('created_at', datetime.utcnow())
                    time_diff = datetime.utcnow() - created_time
                    print(f"   ⚠️ Listing {listing['id'][:8]}... - Status: {status}, Processing for: {time_diff}")
                else:
                    print(f"   📋 Listing {listing['id'][:8]}... - Status: {status}")
            
            print(f"\n   📊 Processing Status Summary:")
            for status, count in processing_statuses.items():
                print(f"      {status}: {count}")
            
            print(f"   📈 Analysis:")
            print(f"      Completed with AI output: {completed_with_output}")
            print(f"      Stuck in processing: {stuck_in_processing}")
            
            return True
            
        except Exception as e:
            print(f"❌ Database query error: {e}")
            return False

    def test_ai_tools_catalog(self):
        """Test AI tools catalog endpoint"""
        print("\n🔍 DEBUGGING STEP 2: Test AI Tools Catalog")
        
        success, response = self.run_test(
            "AI Tools Catalog",
            "GET",
            "ai-tools",
            200
        )
        
        if success and response:
            tools_by_category = response.get('tools_by_category', {})
            total_tools = response.get('total_tools', 0)
            
            print(f"   Total tools available: {total_tools}")
            
            # Check for specific tools mentioned in logs
            all_tools = []
            for category_tools in tools_by_category.values():
                all_tools.extend(category_tools)
            
            tool_names = [tool['name'] for tool in all_tools]
            tool_ids = [tool['id'] for tool in all_tools]
            
            # Look for tools mentioned in the review request
            key_tools = ['listing_luxe_gpt', 'social_snippets_studio', 'comp_cruncher_cma', 'open_house_orchestrator']
            for tool_id in key_tools:
                if tool_id in tool_ids:
                    tool_info = next(tool for tool in all_tools if tool['id'] == tool_id)
                    print(f"   ✅ Found key tool: {tool_info['name']} (ID: {tool_id}, Credits: {tool_info['credits_cost']})")
                else:
                    print(f"   ❌ Missing key tool: {tool_id}")
            
            return True
        
        return success

    def create_test_listing_with_ai_tools(self):
        """Create a test listing with AI tools for processing"""
        print("\n🔍 DEBUGGING STEP 3: Create Test Listing with AI Tools")
        
        if not self.user_token:
            if not self.setup_test_user():
                return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Create listing with the 4 tools mentioned in the review request
        test_data = {
            "property_details": {
                "address": "789 AI Debug Street",
                "city": "San Francisco", 
                "state": "CA",
                "zip_code": "94104",
                "beds": 3,
                "baths": 2.5,
                "sqft": 1800,
                "property_type": "Single Family",
                "listing_price": 1100000
            },
            "description": "Test property for AI processing debug",
            "selected_tool_ids": ["listing_luxe_gpt", "social_snippets_studio", "comp_cruncher_cma", "open_house_orchestrator"],
            "agent_notes": "Debug test - checking AI processing status"
        }
        
        success, response = self.run_test(
            "Create Test Listing with AI Tools",
            "POST",
            "listings",
            200,
            data=test_data,
            headers=headers
        )
        
        if success and response:
            self.test_listing_id = response['id']
            selected_tools = response.get('selected_ai_tools', [])
            total_credits = sum(tool['credits_cost'] for tool in selected_tools)
            
            print(f"   ✅ Test listing created: {self.test_listing_id}")
            print(f"   Selected tools: {[tool['tool_name'] for tool in selected_tools]}")
            print(f"   Total credits cost: {total_credits}")
            print(f"   Initial AI processing status: {response.get('ai_processing_status')}")
            
            return True
        
        return success

    def test_ai_processing_endpoint(self):
        """Test the AI processing endpoint"""
        print("\n🔍 DEBUGGING STEP 4: Test AI Processing Endpoint")
        
        if not self.test_listing_id:
            if not self.create_test_listing_with_ai_tools():
                return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Check user credits before processing
        credits_success, credits_response = self.run_test(
            "Check User Credits Before Processing",
            "GET",
            "auth/credits",
            200,
            headers=headers
        )
        
        if credits_success:
            initial_credits = credits_response.get('credits', 0)
            print(f"   User credits before processing: {initial_credits}")
        
        # Trigger AI processing
        success, response = self.run_test(
            "Trigger AI Processing",
            "POST",
            f"listings/{self.test_listing_id}/process-ai",
            200,
            headers=headers
        )
        
        if success and response:
            print(f"   ✅ AI processing triggered successfully")
            print(f"   Processing response: {json.dumps(response, indent=2)}")
            
            # Check credits after processing
            credits_success, credits_response = self.run_test(
                "Check User Credits After Processing",
                "GET",
                "auth/credits",
                200,
                headers=headers
            )
            
            if credits_success:
                final_credits = credits_response.get('credits', 0)
                credits_used = initial_credits - final_credits
                print(f"   User credits after processing: {final_credits}")
                print(f"   Credits deducted: {credits_used}")
            
            return True
        
        return success

    def monitor_listing_status_updates(self):
        """Monitor listing status updates over time"""
        print("\n🔍 DEBUGGING STEP 5: Monitor Listing Status Updates")
        
        if not self.test_listing_id:
            print("⚠️ No test listing available")
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        # Check status multiple times over a period
        for i in range(5):
            success, response = self.run_test(
                f"Check Listing Status (Attempt {i+1})",
                "GET",
                f"listings/{self.test_listing_id}",
                200,
                headers=headers
            )
            
            if success and response:
                ai_status = response.get('ai_processing_status')
                ai_output = response.get('ai_output')
                updated_at = response.get('updated_at')
                
                print(f"   Status check {i+1}: {ai_status}")
                print(f"   Has AI output: {'Yes' if ai_output else 'No'}")
                print(f"   Last updated: {updated_at}")
                
                if ai_status == 'completed' and ai_output:
                    print(f"   ✅ AI processing completed successfully!")
                    print(f"   AI output keys: {list(ai_output.keys()) if isinstance(ai_output, dict) else 'Not a dict'}")
                    return True
                elif ai_status == 'failed':
                    print(f"   ❌ AI processing failed")
                    return False
            
            if i < 4:  # Don't sleep on the last iteration
                print(f"   Waiting 3 seconds before next check...")
                time.sleep(3)
        
        print(f"   ⚠️ AI processing status monitoring completed")
        return True

    def test_ai_results_endpoint(self):
        """Test the AI results endpoint"""
        print("\n🔍 DEBUGGING STEP 6: Test AI Results Endpoint")
        
        if not self.test_listing_id:
            print("⚠️ No test listing available")
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        success, response = self.run_test(
            "Get AI Results",
            "GET",
            f"listings/{self.test_listing_id}/ai-results",
            200,
            headers=headers
        )
        
        if success and response:
            print(f"   ✅ AI results endpoint accessible")
            print(f"   Results structure: {json.dumps(response, indent=2)[:500]}...")
            return True
        
        return success

    def check_database_consistency(self):
        """Check database consistency for AI processing"""
        print("\n🔍 DEBUGGING STEP 7: Database Consistency Check")
        
        if self.db is None:
            print("⚠️ No database connection available")
            return False
        
        try:
            # Check for listings that should be completed but show wrong status
            inconsistent_listings = list(self.db.listings.find({
                "ai_output": {"$exists": True, "$ne": None},
                "ai_processing_status": {"$ne": "completed"}
            }))
            
            print(f"   Found {len(inconsistent_listings)} listings with AI output but wrong status")
            
            for listing in inconsistent_listings:
                print(f"   ⚠️ Inconsistent listing {listing['id'][:8]}...")
                print(f"      Status: {listing.get('ai_processing_status')}")
                print(f"      Has AI output: {'Yes' if listing.get('ai_output') else 'No'}")
                print(f"      Created: {listing.get('created_at')}")
                print(f"      Updated: {listing.get('updated_at')}")
            
            # Check for listings stuck in processing for too long
            stuck_listings = list(self.db.listings.find({
                "ai_processing_status": "processing",
                "created_at": {"$lt": datetime.utcnow().replace(hour=datetime.utcnow().hour-1)}  # More than 1 hour ago
            }))
            
            print(f"   Found {len(stuck_listings)} listings stuck in processing for >1 hour")
            
            for listing in stuck_listings:
                created_time = listing.get('created_at', datetime.utcnow())
                time_stuck = datetime.utcnow() - created_time
                print(f"   ⚠️ Stuck listing {listing['id'][:8]}... - Processing for: {time_stuck}")
            
            # Check timestamp fields
            if self.test_listing_id:
                test_listing = self.db.listings.find_one({"id": self.test_listing_id})
                if test_listing:
                    print(f"\n   📋 Test listing timestamps:")
                    print(f"      Created: {test_listing.get('created_at')}")
                    print(f"      Updated: {test_listing.get('updated_at')}")
                    if 'completed_at' in test_listing:
                        print(f"      Completed: {test_listing.get('completed_at')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Database consistency check error: {e}")
            return False

    def test_listings_endpoint_status_display(self):
        """Test that GET /api/listings shows correct status"""
        print("\n🔍 DEBUGGING STEP 8: Test Listings Endpoint Status Display")
        
        if not self.user_token:
            print("⚠️ No user token available")
            return False
        
        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        success, response = self.run_test(
            "Get All User Listings",
            "GET",
            "listings",
            200,
            headers=headers
        )
        
        if success and response:
            print(f"   Retrieved {len(response)} listings")
            
            for listing in response:
                listing_id = listing['id'][:8]
                ai_status = listing.get('ai_processing_status', 'unknown')
                has_output = bool(listing.get('ai_output'))
                
                print(f"   📋 Listing {listing_id}... - Status: {ai_status}, Has Output: {has_output}")
                
                # Check for inconsistencies
                if ai_status == 'completed' and not has_output:
                    print(f"   ⚠️ INCONSISTENCY: Listing marked completed but no AI output")
                elif ai_status != 'completed' and has_output:
                    print(f"   ⚠️ INCONSISTENCY: Listing has AI output but not marked completed")
            
            return True
        
        return success

    def run_comprehensive_debug(self):
        """Run comprehensive AI processing debug tests"""
        print("🚀 Starting Comprehensive AI Processing Debug Tests")
        print("=" * 60)
        
        # Step 1: Database analysis
        self.query_database_listings()
        
        # Step 2: Test AI tools catalog
        self.test_ai_tools_catalog()
        
        # Step 3: Create test listing
        self.create_test_listing_with_ai_tools()
        
        # Step 4: Test AI processing
        self.test_ai_processing_endpoint()
        
        # Step 5: Monitor status updates
        self.monitor_listing_status_updates()
        
        # Step 6: Test AI results endpoint
        self.test_ai_results_endpoint()
        
        # Step 7: Database consistency check
        self.check_database_consistency()
        
        # Step 8: Test listings endpoint
        self.test_listings_endpoint_status_display()
        
        # Summary
        print("\n" + "=" * 60)
        print(f"🏁 AI Processing Debug Tests Complete")
        print(f"   Tests run: {self.tests_run}")
        print(f"   Tests passed: {self.tests_passed}")
        print(f"   Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        return self.tests_passed, self.tests_run

if __name__ == "__main__":
    tester = AIProcessingDebugTester()
    passed, total = tester.run_comprehensive_debug()
    
    if passed == total:
        print("\n✅ All AI processing debug tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} AI processing debug tests failed!")
        sys.exit(1)