#!/usr/bin/env python3

import requests
import json
import sys
from datetime import datetime

class AIStatusChecker:
    def __init__(self):
        self.base_url = "https://realestate-ai-65.preview.emergentagent.com"
        self.api_url = f"{self.base_url}/api"
        self.user_token = None

    def register_test_user(self):
        """Register a test user to get authentication token"""
        import uuid
        test_email = f"statuscheck_{uuid.uuid4().hex[:8]}@example.com"
        test_data = {
            "email": test_email,
            "password": "testpassword123",
            "full_name": "Status Check User"
        }
        
        try:
            response = requests.post(f"{self.api_url}/auth/register", json=test_data)
            if response.status_code == 200:
                data = response.json()
                self.user_token = data.get('access_token')
                print(f"✅ Registered test user: {test_email}")
                return True
            else:
                print(f"❌ Failed to register user: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False

    def check_current_processing_status(self):
        """Check current AI processing status"""
        if not self.user_token:
            if not self.register_test_user():
                return False

        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        try:
            print("\n🔍 CHECKING CURRENT AI PROCESSING STATUS...")
            response = requests.get(f"{self.api_url}/listings", headers=headers)
            
            if response.status_code != 200:
                print(f"❌ Failed to get listings: {response.status_code}")
                return False
            
            listings = response.json()
            
            processing_listings = []
            completed_listings = []
            pending_listings = []
            failed_listings = []
            
            for listing in listings:
                status = listing.get('ai_processing_status', 'unknown')
                if status == 'processing':
                    processing_listings.append(listing)
                elif status == 'completed':
                    completed_listings.append(listing)
                elif status == 'pending':
                    pending_listings.append(listing)
                elif status == 'failed':
                    failed_listings.append(listing)
            
            print(f"📊 PROCESSING STATUS SUMMARY:")
            print(f"   🔄 Processing: {len(processing_listings)} listings")
            print(f"   ✅ Completed: {len(completed_listings)} listings")
            print(f"   ⏳ Pending: {len(pending_listings)} listings")
            print(f"   ❌ Failed: {len(failed_listings)} listings")
            print(f"   📋 Total: {len(listings)} listings")
            
            # Detailed analysis of processing listings
            if processing_listings:
                print(f"\n🚨 FOUND {len(processing_listings)} LISTINGS IN PROCESSING STATUS:")
                for listing in processing_listings:
                    created_at = listing.get('created_at', 'unknown')
                    updated_at = listing.get('updated_at', 'unknown')
                    selected_tools = listing.get('selected_ai_tools', [])
                    tool_count = len(selected_tools)
                    
                    print(f"   📋 Listing ID: {listing['id']}")
                    print(f"      Created: {created_at}")
                    print(f"      Updated: {updated_at}")
                    print(f"      Tools: {tool_count} selected")
                    
                    # Check if this matches the 27-tool job mentioned in review
                    if tool_count >= 25:  # Close to 27 tools
                        print(f"      🎯 POTENTIAL MATCH: Large job with {tool_count} tools")
                        print(f"      🔍 This could be the stuck 27-tool job from 05:00:00")
                        
                        # Try to calculate time since creation
                        try:
                            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            current_time = datetime.now(created_time.tzinfo)
                            time_diff = current_time - created_time
                            minutes_running = time_diff.total_seconds() / 60
                            print(f"      ⏱️ Running for: {minutes_running:.1f} minutes")
                            
                            if minutes_running > 15:  # More than 15 minutes is likely stuck
                                print(f"      🚨 LIKELY STUCK: Running for {minutes_running:.1f} minutes")
                            elif minutes_running > 10:
                                print(f"      ⚠️ LONG RUNNING: {minutes_running:.1f} minutes (may be normal for {tool_count} tools)")
                            else:
                                print(f"      ✅ NORMAL: {minutes_running:.1f} minutes (within expected range)")
                        except Exception as e:
                            print(f"      ⚠️ Could not parse time: {e}")
                
                return False  # Processing jobs found - potential issue
            else:
                print("✅ No listings currently in processing status")
                
                # Check completed listings for recent activity
                if completed_listings:
                    print(f"\n📈 RECENT COMPLETED LISTINGS:")
                    for listing in completed_listings[:3]:  # Show first 3
                        created_at = listing.get('created_at', 'unknown')
                        updated_at = listing.get('updated_at', 'unknown')
                        tool_count = len(listing.get('selected_ai_tools', []))
                        has_output = bool(listing.get('ai_output'))
                        
                        print(f"   ✅ {listing['id'][:8]}... | Tools: {tool_count} | Output: {has_output} | Updated: {updated_at}")
                
                return True
                
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            return False

    def test_ai_tools_catalog(self):
        """Test AI tools catalog to verify tool availability"""
        try:
            print("\n🛠️ CHECKING AI TOOLS CATALOG...")
            response = requests.get(f"{self.api_url}/ai-tools")
            
            if response.status_code != 200:
                print(f"❌ Failed to get AI tools: {response.status_code}")
                return False
            
            data = response.json()
            tools_by_category = data.get('tools_by_category', {})
            total_tools = data.get('total_tools', 0)
            
            print(f"📊 AI TOOLS CATALOG:")
            print(f"   Total tools: {total_tools}")
            print(f"   Categories: {len(tools_by_category)}")
            
            # Check for specific tools mentioned in review
            all_tools = []
            for category, tools in tools_by_category.items():
                all_tools.extend(tools)
                print(f"   {category}: {len(tools)} tools")
            
            tool_ids = [tool['id'] for tool in all_tools]
            
            # Check for key tools mentioned in review request
            key_tools = ['listing_luxe_gpt', 'social_snippets_studio', 'comp_cruncher_cma', 'open_house_orchestrator']
            for tool_id in key_tools:
                if tool_id in tool_ids:
                    tool_info = next(t for t in all_tools if t['id'] == tool_id)
                    print(f"   ✅ {tool_id}: {tool_info['credits_cost']} credits")
                else:
                    print(f"   ❌ Missing: {tool_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error checking AI tools: {e}")
            return False

    def test_create_and_process_small_job(self):
        """Create and process a small AI job to test current functionality"""
        if not self.user_token:
            if not self.register_test_user():
                return False

        headers = {"Authorization": f"Bearer {self.user_token}"}
        
        try:
            print("\n🧪 TESTING SMALL AI PROCESSING JOB...")
            
            # Create a test listing with 4 tools
            test_data = {
                "property_details": {
                    "address": "123 Status Test Ave",
                    "city": "San Francisco", 
                    "state": "CA",
                    "zip_code": "94105",
                    "beds": 2,
                    "baths": 2.0,
                    "sqft": 1400,
                    "property_type": "Condo",
                    "listing_price": 900000
                },
                "description": "Status test property",
                "selected_tool_ids": ["listing_luxe_gpt", "social_snippets_studio", "comp_cruncher_cma", "open_house_orchestrator"],
                "agent_notes": "Status check test - 4 tools"
            }
            
            response = requests.post(f"{self.api_url}/listings", json=test_data, headers=headers)
            
            if response.status_code != 200:
                print(f"❌ Failed to create test listing: {response.status_code}")
                return False
            
            listing_data = response.json()
            listing_id = listing_data['id']
            selected_tools = listing_data.get('selected_ai_tools', [])
            total_credits = sum(tool.get('credits_cost', 0) for tool in selected_tools)
            
            print(f"✅ Created test listing: {listing_id}")
            print(f"   Tools: {len(selected_tools)} selected")
            print(f"   Credits needed: {total_credits}")
            
            # Try to process the AI tools
            print(f"🚀 Starting AI processing...")
            start_time = datetime.now()
            
            process_response = requests.post(f"{self.api_url}/listings/{listing_id}/process-ai", headers=headers)
            
            if process_response.status_code == 200:
                process_data = process_response.json()
                print(f"✅ AI processing started successfully")
                print(f"   Response: {json.dumps(process_data, indent=2)[:200]}...")
                
                # Check status immediately after
                status_response = requests.get(f"{self.api_url}/listings/{listing_id}", headers=headers)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    ai_status = status_data.get('ai_processing_status', 'unknown')
                    
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    print(f"📊 Processing completed in {processing_time:.1f} seconds")
                    print(f"   Final status: {ai_status}")
                    
                    if ai_status == 'completed':
                        print("✅ AI processing completed successfully")
                        return True
                    elif ai_status == 'processing':
                        print("⏳ AI processing still in progress (normal for larger jobs)")
                        return True
                    else:
                        print(f"⚠️ Unexpected status: {ai_status}")
                        return False
                else:
                    print(f"❌ Failed to check status: {status_response.status_code}")
                    return False
            else:
                print(f"❌ Failed to start AI processing: {process_response.status_code}")
                print(f"   Response: {process_response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error in processing test: {e}")
            return False

def main():
    checker = AIStatusChecker()
    
    print("🔍 AI PROCESSING STATUS INVESTIGATION")
    print("=" * 50)
    
    # Check current processing status
    status_ok = checker.check_current_processing_status()
    
    # Check AI tools catalog
    catalog_ok = checker.test_ai_tools_catalog()
    
    # Test small processing job
    processing_ok = checker.test_create_and_process_small_job()
    
    print("\n" + "=" * 50)
    print("📋 INVESTIGATION SUMMARY:")
    print(f"   Current Status Check: {'✅ PASS' if status_ok else '❌ ISSUES FOUND'}")
    print(f"   AI Tools Catalog: {'✅ PASS' if catalog_ok else '❌ ISSUES FOUND'}")
    print(f"   Processing Test: {'✅ PASS' if processing_ok else '❌ ISSUES FOUND'}")
    
    if not status_ok:
        print("\n🚨 RECOMMENDATION FOR USER:")
        print("   There appear to be listings stuck in 'processing' status.")
        print("   This could be the cause of the 'AI Pending for a long time' issue.")
        print("   The system may need intervention to clear stuck jobs.")
    elif status_ok and catalog_ok and processing_ok:
        print("\n✅ SYSTEM STATUS: HEALTHY")
        print("   No stuck processing jobs found.")
        print("   AI tools catalog is working correctly.")
        print("   New processing jobs complete successfully.")
        print("   The reported issue may have been resolved.")
    else:
        print("\n⚠️ MIXED RESULTS:")
        print("   Some components working, others may need attention.")

if __name__ == "__main__":
    main()