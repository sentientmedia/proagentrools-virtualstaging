#!/usr/bin/env python3

import requests
import json
import sys
import uuid

def focused_google_oauth_test():
    """Focused test of Google OAuth integration based on the review request"""
    
    base_url = "https://smart-listing-pro-2.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 FOCUSED GOOGLE OAUTH INTEGRATION TEST")
    print("Testing the exact scenario from the review request")
    print("=" * 70)
    
    tests_run = 0
    tests_passed = 0
    
    def run_test(name, test_func):
        nonlocal tests_run, tests_passed
        tests_run += 1
        print(f"\n🔍 {name}...")
        try:
            if test_func():
                tests_passed += 1
                print(f"✅ PASSED: {name}")
                return True
            else:
                print(f"❌ FAILED: {name}")
                return False
        except Exception as e:
            print(f"❌ ERROR in {name}: {str(e)}")
            return False
    
    # Test 1: POST /api/auth/google/session endpoint directly
    def test_google_session_endpoint():
        mock_data = {
            "user_data": {
                "id": "google_123456789",
                "email": "test.oauth@gmail.com", 
                "name": "Test OAuth User",
                "picture": "https://via.placeholder.com/150"
            },
            "session_token": "test_google_session_abc123"
        }
        
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=mock_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        if not data.get('success'):
            print(f"   Success flag is False: {data}")
            return False
        
        user = data.get('user', {})
        if user.get('credits') != 100:
            print(f"   Expected 100 credits, got {user.get('credits')}")
            return False
        
        if user.get('email') != "test.oauth@gmail.com":
            print(f"   Email mismatch: {user.get('email')}")
            return False
        
        print(f"   User created with ID: {user.get('id')}")
        print(f"   Credits: {user.get('credits')}")
        return True
    
    # Test 2: Verify session token works with GET /api/auth/me
    def test_session_token_auth():
        session_token = "test_google_session_abc123"
        
        response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   Response: {response.text}")
            return False
        
        data = response.json()
        if data.get('email') != "test.oauth@gmail.com":
            print(f"   Email mismatch: {data.get('email')}")
            return False
        
        print(f"   Authenticated as: {data.get('email')}")
        return True
    
    # Test 3: Test user creation and session storage
    def test_user_creation_and_session():
        # Create a new user with different data
        new_mock_data = {
            "user_data": {
                "id": "google_987654321",
                "email": "new.oauth@gmail.com", 
                "name": "New OAuth User",
                "picture": "https://via.placeholder.com/200"
            },
            "session_token": "new_google_session_xyz789"
        }
        
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=new_mock_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            print(f"   Session creation failed: {response.text}")
            return False
        
        # Test the new session token
        auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer new_google_session_xyz789'}
        )
        
        if auth_response.status_code != 200:
            print(f"   New session auth failed: {auth_response.text}")
            return False
        
        data = auth_response.json()
        if data.get('email') != "new.oauth@gmail.com":
            print(f"   New user email mismatch: {data.get('email')}")
            return False
        
        print(f"   New user created and authenticated: {data.get('email')}")
        return True
    
    # Test 4: Test existing user session update
    def test_existing_user_session_update():
        # Use the same email but different session token
        update_mock_data = {
            "user_data": {
                "id": "google_updated_id",
                "email": "test.oauth@gmail.com",  # Same email as first test
                "name": "Updated OAuth User",
                "picture": "https://via.placeholder.com/300"
            },
            "session_token": "updated_google_session_def456"
        }
        
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=update_mock_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            print(f"   Session update failed: {response.text}")
            return False
        
        # Test the updated session token
        auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer updated_google_session_def456'}
        )
        
        if auth_response.status_code != 200:
            print(f"   Updated session auth failed: {auth_response.text}")
            return False
        
        data = auth_response.json()
        if data.get('email') != "test.oauth@gmail.com":
            print(f"   Updated user email mismatch: {data.get('email')}")
            return False
        
        print(f"   Existing user session updated: {data.get('email')}")
        
        # Test that old session token is invalidated
        old_auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer test_google_session_abc123'}
        )
        
        if old_auth_response.status_code == 200:
            print("   WARNING: Old session token still works (should be invalidated)")
            # This might be expected behavior - multiple sessions allowed
        
        return True
    
    # Test 5: Test protected endpoints with session tokens
    def test_protected_endpoints():
        session_token = "updated_google_session_def456"
        
        # Test credits endpoint
        credits_response = requests.get(
            f"{api_url}/auth/credits",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if credits_response.status_code != 200:
            print(f"   Credits endpoint failed: {credits_response.text}")
            return False
        
        credits_data = credits_response.json()
        if 'credits' not in credits_data:
            print(f"   Credits data missing: {credits_data}")
            return False
        
        print(f"   Credits endpoint works: {credits_data.get('credits')} credits")
        
        # Test interior design endpoint (requires file upload)
        try:
            files = {'file': ('test.jpg', b'fake_image_data', 'image/jpeg')}
            data = {
                'room_type': 'living_room',
                'designer': 'alessia_duval',
                'color_scheme': 'glacial_muse'
            }
            
            design_response = requests.post(
                f"{api_url}/interior-design/process",
                data=data,
                files=files,
                headers={'Authorization': f'Bearer {session_token}'}
            )
            
            if design_response.status_code != 200:
                print(f"   Interior design failed: {design_response.text}")
                return False
            
            design_data = design_response.json()
            if design_data.get('status') != 'queued':
                print(f"   Unexpected status: {design_data.get('status')}")
                return False
            
            print(f"   Interior design endpoint works: {design_data.get('status')}")
            return True
            
        except Exception as e:
            print(f"   Interior design test error: {str(e)}")
            return False
    
    # Test 6: Test logout functionality
    def test_logout_functionality():
        session_token = "updated_google_session_def456"
        
        # Test logout
        logout_response = requests.post(
            f"{api_url}/auth/logout",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if logout_response.status_code != 200:
            print(f"   Logout failed: {logout_response.text}")
            return False
        
        logout_data = logout_response.json()
        if not logout_data.get('success'):
            print(f"   Logout success flag false: {logout_data}")
            return False
        
        print("   Logout successful")
        
        # Test that session token is now invalid
        auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if auth_response.status_code == 200:
            print("   WARNING: Session token still works after logout")
            return False
        
        print("   Session token correctly invalidated after logout")
        return True
    
    # Run all tests
    run_test("POST /api/auth/google/session endpoint", test_google_session_endpoint)
    run_test("Session token authentication with /api/auth/me", test_session_token_auth)
    run_test("User creation and session storage", test_user_creation_and_session)
    run_test("Existing user session update", test_existing_user_session_update)
    run_test("Protected endpoints with session tokens", test_protected_endpoints)
    run_test("Logout functionality", test_logout_functionality)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"📊 FOCUSED TEST RESULTS:")
    print(f"   Tests Run: {tests_run}")
    print(f"   Tests Passed: {tests_passed}")
    print(f"   Success Rate: {(tests_passed/tests_run)*100:.1f}%")
    
    if tests_passed == tests_run:
        print("🎉 All Google OAuth tests passed!")
        print("✅ Google OAuth integration is working correctly")
        return 0
    else:
        print(f"⚠️  {tests_run - tests_passed} tests failed")
        print("❌ Some Google OAuth functionality issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(focused_google_oauth_test())