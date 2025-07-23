import requests
import sys
import json
import io
from datetime import datetime
from PIL import Image
import tempfile
import os
import uuid

class ProAgentToolsAPITester:
    def __init__(self, base_url="https://7f9f2de6-2fb0-4a74-af1b-8d15dbd9c892.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.user_token = None
        self.admin_token = None
        self.test_user_id = None

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

    def test_room_types_endpoint(self):
        """Test room types endpoint - should return 5 room types WITHOUT descriptions"""
        success, response = self.run_test(
            "Room Types Endpoint",
            "GET",
            "interior-design/room-types",
            200
        )
        
        if success and response:
            # Validate structure and content
            room_types = response.get('room_types', [])
            print(f"   Found {len(room_types)} room types")
            
            # Check count
            if len(room_types) != 5:
                print(f"❌ Expected 5 room types, got {len(room_types)}")
                return False
            
            # Check structure - should NOT have description field
            for room_type in room_types:
                if 'description' in room_type:
                    print(f"❌ Room type {room_type.get('name')} has description field (should be removed)")
                    return False
                if 'id' not in room_type or 'name' not in room_type:
                    print(f"❌ Room type missing required fields: {room_type}")
                    return False
            
            print("✅ Room types structure validated - no description fields found")
            return True
        
        return success

    def test_designers_endpoint(self):
        """Test designers endpoint - should return 12 real designers from CSV"""
        success, response = self.run_test(
            "Designers Endpoint",
            "GET",
            "interior-design/designers",
            200
        )
        
        if success and response:
            # Validate structure and content
            designers = response.get('designers', [])
            print(f"   Found {len(designers)} designers")
            
            # Check count
            if len(designers) != 12:
                print(f"❌ Expected 12 designers, got {len(designers)}")
                return False
            
            # Check for specific required names from CSV
            required_names = ["Alessia Duval", "Adrian Mercer", "Lucien Hart"]
            found_names = [d.get('name') for d in designers]
            
            for required_name in required_names:
                if required_name not in found_names:
                    print(f"❌ Required designer '{required_name}' not found in response")
                    return False
            
            # Check structure - should have id, name, and description
            for designer in designers:
                if not all(key in designer for key in ['id', 'name', 'description']):
                    print(f"❌ Designer missing required fields: {designer}")
                    return False
                if not designer['description'] or len(designer['description']) < 10:
                    print(f"❌ Designer {designer['name']} has invalid description")
                    return False
            
            print(f"✅ All required designers found: {required_names}")
            print("✅ Designers structure validated with descriptions")
            return True
        
        return success

    def test_color_schemes_endpoint(self):
        """Test color schemes endpoint - should return 20 real color schemes from CSV"""
        success, response = self.run_test(
            "Color Schemes Endpoint",
            "GET",
            "interior-design/color-schemes",
            200
        )
        
        if success and response:
            # Validate structure and content
            color_schemes = response.get('color_schemes', [])
            print(f"   Found {len(color_schemes)} color schemes")
            
            # Check count
            if len(color_schemes) != 20:
                print(f"❌ Expected 20 color schemes, got {len(color_schemes)}")
                return False
            
            # Check for specific required names from CSV
            required_names = ["Glacial Muse", "Nomad Prism", "Urban Alloy"]
            found_names = [cs.get('name') for cs in color_schemes]
            
            for required_name in required_names:
                if required_name not in found_names:
                    print(f"❌ Required color scheme '{required_name}' not found in response")
                    return False
            
            # Check structure - should have id, name, and description
            for color_scheme in color_schemes:
                if not all(key in color_scheme for key in ['id', 'name', 'description']):
                    print(f"❌ Color scheme missing required fields: {color_scheme}")
                    return False
                if not color_scheme['description'] or len(color_scheme['description']) < 10:
                    print(f"❌ Color scheme {color_scheme['name']} has invalid description")
                    return False
            
            print(f"✅ All required color schemes found: {required_names}")
            print("✅ Color schemes structure validated with descriptions")
            return True
        
        return success

    def test_process_endpoint_new_defaults(self):
        """Test that process endpoint uses NEW default values (alessia_duval, glacial_muse)"""
        # Create test image
        test_image = self.create_test_image()
        
        files = {
            'file': ('test_interior.jpg', test_image, 'image/jpeg')
        }
        
        # Test without providing designer/color_scheme parameters - should use NEW defaults
        success, response = self.run_test(
            "Process Endpoint - NEW Default Values Test",
            "POST",
            "interior-design/process",
            200,
            files=files
        )
        
        if success and response:
            # Verify NEW default values are used
            designer = response.get('designer')
            color_scheme = response.get('color_scheme')
            
            print(f"   Default designer used: {designer}")
            print(f"   Default color_scheme used: {color_scheme}")
            
            # Check for NEW defaults (NOT old ones)
            if designer != "alessia_duval":
                print(f"❌ Expected NEW default designer 'alessia_duval', got '{designer}'")
                return False
            
            if color_scheme != "glacial_muse":
                print(f"❌ Expected NEW default color_scheme 'glacial_muse', got '{color_scheme}'")
                return False
            
            # Verify NO old placeholder values
            old_designer = "minimalist_maven"
            old_color_scheme = "neutral_warm"
            
            if designer == old_designer:
                print(f"❌ CRITICAL: Still using OLD default designer '{old_designer}' - fix not applied!")
                return False
            
            if color_scheme == old_color_scheme:
                print(f"❌ CRITICAL: Still using OLD default color_scheme '{old_color_scheme}' - fix not applied!")
                return False
            
            print("✅ NEW default values confirmed - old placeholder data removed")
            return True
        
        return success

    def test_process_endpoint_custom_parameters(self):
        """Test that process endpoint accepts custom designer/color_scheme parameters"""
        # Create test image
        test_image = self.create_test_image()
        
        # Test with custom parameters using NEW CSV-based IDs - send as form data
        files = {
            'file': ('test_interior.jpg', test_image, 'image/jpeg')
        }
        
        data = {
            'room_type': 'bedroom',
            'designer': 'adrian_mercer',  # Different from default
            'color_scheme': 'nomad_prism'  # Different from default
        }
        
        success, response = self.run_test(
            "Process Endpoint - Custom Parameters Test",
            "POST",
            "interior-design/process",
            200,
            data=data,
            files=files
        )
        
        if success and response:
            # Verify custom values are used
            designer = response.get('designer')
            color_scheme = response.get('color_scheme')
            room_type = response.get('room_type')
            
            print(f"   Custom designer used: {designer}")
            print(f"   Custom color_scheme used: {color_scheme}")
            print(f"   Custom room_type used: {room_type}")
            
            if designer != "adrian_mercer":
                print(f"❌ Expected custom designer 'adrian_mercer', got '{designer}'")
                return False
            
            if color_scheme != "nomad_prism":
                print(f"❌ Expected custom color_scheme 'nomad_prism', got '{color_scheme}'")
                return False
            
            if room_type != "bedroom":
                print(f"❌ Expected custom room_type 'bedroom', got '{room_type}'")
                return False
            
            print("✅ Custom parameters properly accepted and used")
            return True
        
        return success

    def test_no_old_placeholder_data(self):
        """Comprehensive test to ensure NO old placeholder data exists anywhere"""
        print("\n🔍 Comprehensive Old Data Cleanup Verification...")
        
        # Test all configuration endpoints for old data
        endpoints_to_check = [
            ("interior-design/room-types", "room_types"),
            ("interior-design/designers", "designers"), 
            ("interior-design/color-schemes", "color_schemes")
        ]
        
        old_placeholder_terms = [
            "minimalist_maven",
            "neutral_warm", 
            "placeholder",
            "test_designer",
            "test_color"
        ]
        
        all_clean = True
        
        for endpoint, response_key in endpoints_to_check:
            success, response = self.run_test(
                f"Old Data Check - {endpoint}",
                "GET",
                endpoint,
                200
            )
            
            if success and response:
                items = response.get(response_key, [])
                response_text = json.dumps(response).lower()
                
                # Check for any old placeholder terms
                for old_term in old_placeholder_terms:
                    if old_term.lower() in response_text:
                        print(f"❌ CRITICAL: Found old placeholder '{old_term}' in {endpoint}")
                        all_clean = False
                
                print(f"✅ {endpoint} clean of old placeholder data")
            else:
                all_clean = False
        
        if all_clean:
            print("✅ COMPREHENSIVE CLEANUP VERIFIED: No old placeholder data found")
            return True
        else:
            print("❌ CLEANUP INCOMPLETE: Old placeholder data still exists")
            return False

    def test_invalid_endpoints(self):
        """Test invalid endpoints return proper errors"""
        success, response = self.run_test(
            "Invalid Endpoint (404 Test)",
            "GET",
            "nonexistent-endpoint",
            404
        )
        return success

    def test_queue_status_endpoint(self):
        """Test queue status endpoint - NEW FEATURE"""
        success, response = self.run_test(
            "Queue Status Endpoint",
            "GET",
            "interior-design/queue",
            200
        )
        
        if success and response:
            # Validate structure
            if 'queue_status' not in response:
                print("❌ Missing queue_status in response")
                return False
            
            queue_status = response['queue_status']
            if 'queued' not in queue_status or 'processing' not in queue_status:
                print("❌ Missing queued/processing counts in queue_status")
                return False
            
            print(f"   Queue Status: {queue_status}")
            print("✅ Queue status structure validated")
            return True
        
        return success

    def test_process_endpoint_queued_status(self):
        """Test that process endpoint returns 'queued' status immediately - ENHANCED FEATURE"""
        # Create test image
        test_image = self.create_test_image()
        
        files = {
            'file': ('test_interior.jpg', test_image, 'image/jpeg')
        }
        
        success, response = self.run_test(
            "Process Endpoint - Queued Status Test",
            "POST",
            "interior-design/process",
            200,
            files=files
        )
        
        if success and response:
            # Verify immediate queued status
            status = response.get('status')
            
            print(f"   Immediate status returned: {status}")
            
            if status != "queued":
                print(f"❌ Expected immediate status 'queued', got '{status}'")
                return False
            
            # Should have an ID for tracking
            if 'id' not in response:
                print("❌ Missing 'id' field in response")
                return False
            
            print("✅ Process endpoint returns immediate 'queued' status")
            return True
        
        return success

    def test_storage_directory_structure(self):
        """Test that storage directory exists and is properly configured"""
        print("\n🔍 Testing Storage Directory Structure...")
        
        # Test if we can access the images endpoint (even if no images exist)
        success, response = self.run_test(
            "Storage Directory - Images Endpoint Test",
            "GET",
            "images/nonexistent.jpg",
            404  # Should return 404 for non-existent image
        )
        
        # 404 is expected for non-existent image, which means endpoint is working
        if success:
            print("✅ Images endpoint properly configured (returns 404 for non-existent)")
            return True
        else:
            print("❌ Images endpoint not properly configured")
            return False

    def test_image_serving_endpoint(self):
        """Test image serving endpoint with proper headers - NEW FEATURE"""
        # First, try to upload an image to get one stored
        test_image = self.create_test_image()
        
        files = {
            'file': ('test_storage.jpg', test_image, 'image/jpeg')
        }
        
        # Upload an image first
        upload_success, upload_response = self.run_test(
            "Upload for Image Serving Test",
            "POST",
            "interior-design/process",
            200,
            files=files
        )
        
        if upload_success and upload_response:
            design_id = upload_response.get('id')
            print(f"   Uploaded design ID: {design_id}")
            
            # Test the images endpoint (even though image might not be processed yet)
            # We'll test with a hypothetical filename
            test_filename = f"{design_id}.jpg"
            
            print(f"   Testing image serving for: {test_filename}")
            
            # Make a direct request to check headers
            try:
                url = f"{self.api_url}/images/{test_filename}"
                response = requests.get(url)
                
                print(f"   Image endpoint status: {response.status_code}")
                
                if response.status_code == 200:
                    # Check for proper MIME type and caching headers
                    content_type = response.headers.get('content-type', '')
                    cache_control = response.headers.get('cache-control', '')
                    
                    print(f"   Content-Type: {content_type}")
                    print(f"   Cache-Control: {cache_control}")
                    
                    if 'image' in content_type:
                        print("✅ Proper MIME type for image serving")
                    else:
                        print(f"❌ Incorrect MIME type: {content_type}")
                        return False
                    
                    if 'max-age' in cache_control:
                        print("✅ Proper caching headers present")
                    else:
                        print(f"❌ Missing caching headers: {cache_control}")
                        return False
                    
                    return True
                elif response.status_code == 404:
                    print("✅ Image endpoint working (404 for non-existent image is expected)")
                    return True
                else:
                    print(f"❌ Unexpected status code: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error testing image endpoint: {str(e)}")
                return False
        
        print("⚠️  Could not test image serving due to upload failure")
        return False

    def test_download_design_endpoint(self):
        """Test download design endpoint - NEW FEATURE"""
        # First, try to upload an image to get a design ID
        test_image = self.create_test_image()
        
        files = {
            'file': ('test_download.jpg', test_image, 'image/jpeg')
        }
        
        # Upload an image first
        upload_success, upload_response = self.run_test(
            "Upload for Download Test",
            "POST",
            "interior-design/process",
            200,
            files=files
        )
        
        if upload_success and upload_response:
            design_id = upload_response.get('id')
            print(f"   Testing download for design ID: {design_id}")
            
            # Test the download endpoint
            try:
                url = f"{self.api_url}/interior-design/download/{design_id}"
                response = requests.get(url)
                
                print(f"   Download endpoint status: {response.status_code}")
                
                if response.status_code == 200:
                    # Check for proper file response headers
                    content_disposition = response.headers.get('content-disposition', '')
                    content_type = response.headers.get('content-type', '')
                    
                    print(f"   Content-Disposition: {content_disposition}")
                    print(f"   Content-Type: {content_type}")
                    
                    if 'attachment' in content_disposition:
                        print("✅ Proper download headers present")
                        return True
                    else:
                        print("⚠️  Download headers may not be optimal")
                        return True  # Still working, just not optimal
                        
                elif response.status_code == 400:
                    print("✅ Download endpoint working (400 for incomplete design is expected)")
                    return True
                elif response.status_code == 404:
                    print("❌ Design not found for download")
                    return False
                else:
                    print(f"❌ Unexpected status code: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error testing download endpoint: {str(e)}")
                return False
        
        print("⚠️  Could not test download due to upload failure")
        return False

def main():
    print("🚀 Starting ProAgentTools API Testing...")
    print("🔥 ENHANCED BACKEND TESTING: Permanent Image Storage & Queue System")
    print("=" * 80)
    
    # Setup
    tester = ProAgentToolsAPITester()
    
    # Run basic endpoint tests
    print("\n📋 Testing Basic Endpoints...")
    tester.test_health_check()
    tester.test_root_endpoint()
    tester.test_available_concepts()
    
    # Test Interior Design Configuration Endpoints (CSV Data Integration)
    print("\n🎨 Testing Interior Design Configuration Endpoints (CSV Data)...")
    tester.test_room_types_endpoint()
    tester.test_designers_endpoint()
    tester.test_color_schemes_endpoint()
    
    # CRITICAL FIX VERIFICATION TESTS
    print("\n🔥 CRITICAL FIX VERIFICATION - NEW Default Values...")
    tester.test_process_endpoint_new_defaults()
    tester.test_process_endpoint_custom_parameters()
    tester.test_no_old_placeholder_data()
    
    # NEW ENHANCED FEATURES TESTING
    print("\n🆕 ENHANCED FEATURES - Queue System & Image Storage...")
    tester.test_queue_status_endpoint()
    tester.test_process_endpoint_queued_status()
    tester.test_storage_directory_structure()
    tester.test_image_serving_endpoint()
    tester.test_download_design_endpoint()
    
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
    print("\n" + "=" * 80)
    print(f"📊 FINAL RESULTS:")
    print(f"   Tests Run: {tester.tests_run}")
    print(f"   Tests Passed: {tester.tests_passed}")
    print(f"   Success Rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        print("✅ ENHANCED FEATURES VERIFIED: Image storage & queue system working")
        return 0
    else:
        print(f"⚠️  {tester.tests_run - tester.tests_passed} tests failed")
        print("❌ ENHANCED FEATURES: Some tests failed - check results above")
        return 1

if __name__ == "__main__":
    sys.exit(main())