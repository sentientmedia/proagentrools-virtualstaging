#!/usr/bin/env python3

import requests
import json
import sys

def test_google_oauth_debug():
    """Debug Google OAuth integration with the specific mock data from the review request"""
    
    base_url = "https://realestate-ai-65.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 DEBUGGING GOOGLE OAUTH INTEGRATION")
    print("=" * 60)
    
    # Test data from the review request
    mock_google_session_data = {
        "user_data": {
            "id": "google_123456789",
            "email": "test.oauth@gmail.com", 
            "name": "Test OAuth User",
            "picture": "https://via.placeholder.com/150"
        },
        "session_token": "test_google_session_abc123"
    }
    
    print("1. Testing POST /api/auth/google/session with mock data...")
    print(f"   Mock data: {json.dumps(mock_google_session_data, indent=2)}")
    
    # Test the Google OAuth session endpoint
    try:
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=mock_google_session_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('success'):
                print("✅ Google OAuth session created successfully")
                user_data = response_data.get('user', {})
                print(f"   User ID: {user_data.get('id')}")
                print(f"   Email: {user_data.get('email')}")
                print(f"   Credits: {user_data.get('credits')}")
                
                # Store session token for next test
                session_token = mock_google_session_data["session_token"]
                
                print("\n2. Testing GET /api/auth/me with session token...")
                
                # Test authentication with the session token
                auth_response = requests.get(
                    f"{api_url}/auth/me",
                    headers={'Authorization': f'Bearer {session_token}'}
                )
                
                print(f"   Status Code: {auth_response.status_code}")
                print(f"   Response: {auth_response.text}")
                
                if auth_response.status_code == 200:
                    print("✅ Session token authentication working")
                    auth_data = auth_response.json()
                    print(f"   Authenticated as: {auth_data.get('email')}")
                else:
                    print("❌ Session token authentication failed")
                    print("   This is the core issue - session tokens not working for authentication")
                
                print("\n3. Testing GET /api/auth/credits with session token...")
                
                # Test credits endpoint with session token
                credits_response = requests.get(
                    f"{api_url}/auth/credits",
                    headers={'Authorization': f'Bearer {session_token}'}
                )
                
                print(f"   Status Code: {credits_response.status_code}")
                print(f"   Response: {credits_response.text}")
                
                if credits_response.status_code == 200:
                    print("✅ Credits endpoint working with session token")
                else:
                    print("❌ Credits endpoint failed with session token")
                
                print("\n4. Testing protected endpoint access...")
                
                # Test a protected endpoint that requires credits
                test_image_data = {
                    'room_type': 'living_room',
                    'designer': 'alessia_duval',
                    'color_scheme': 'glacial_muse'
                }
                
                # Create a simple test file
                files = {'file': ('test.jpg', b'fake_image_data', 'image/jpeg')}
                
                design_response = requests.post(
                    f"{api_url}/interior-design/process",
                    data=test_image_data,
                    files=files,
                    headers={'Authorization': f'Bearer {session_token}'}
                )
                
                print(f"   Interior Design Status Code: {design_response.status_code}")
                print(f"   Interior Design Response: {design_response.text[:200]}...")
                
                if design_response.status_code == 200:
                    print("✅ Protected endpoint working with session token")
                else:
                    print("❌ Protected endpoint failed with session token")
                
            else:
                print("❌ Google OAuth session creation failed - success=false")
        else:
            print("❌ Google OAuth session endpoint failed")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY:")
    print("- Google OAuth session creation: Check if POST /api/auth/google/session works")
    print("- Session token storage: Check if session is stored in user_sessions collection")
    print("- Session token authentication: Check if get_current_user_enhanced works")
    print("- Session expiry: Check if session expiry validation works")
    print("- Database integration: Check MongoDB user_sessions collection")

if __name__ == "__main__":
    test_google_oauth_debug()