#!/usr/bin/env python3

import requests
import json
import sys
import uuid

def investigate_session_token_issue():
    """Investigate why session tokens are failing in the backend_test.py"""
    
    base_url = "https://realestate-ai-65.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 INVESTIGATING SESSION TOKEN ISSUE")
    print("=" * 60)
    
    # Create a session similar to backend_test.py
    google_user_data = {
        "id": f"google_user_{uuid.uuid4().hex[:8]}",
        "email": f"googleuser_{uuid.uuid4().hex[:8]}@gmail.com",
        "name": "Google Test User",
        "picture": "https://lh3.googleusercontent.com/test-avatar"
    }
    
    session_token = f"google_session_{uuid.uuid4().hex}"
    
    test_data = {
        "user_data": google_user_data,
        "session_token": session_token
    }
    
    print("1. Creating Google OAuth session...")
    print(f"   Session token: {session_token}")
    
    # Create the session
    try:
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ Session created successfully")
            
            # Immediately test the session token
            print("\n2. Testing session token immediately after creation...")
            
            auth_response = requests.get(
                f"{api_url}/auth/me",
                headers={'Authorization': f'Bearer {session_token}'}
            )
            
            print(f"   Status Code: {auth_response.status_code}")
            print(f"   Response: {auth_response.text}")
            
            if auth_response.status_code == 200:
                print("✅ Session token works immediately after creation")
            else:
                print("❌ Session token fails immediately after creation")
                print("   This suggests an issue with session storage or retrieval")
                
                # Let's check what happens with a different session token format
                print("\n3. Testing with different session token format...")
                
                simple_token = "test_simple_token_123"
                simple_data = {
                    "user_data": {
                        "id": "google_simple_test",
                        "email": "simple@test.com",
                        "name": "Simple Test User",
                        "picture": "https://via.placeholder.com/150"
                    },
                    "session_token": simple_token
                }
                
                simple_response = requests.post(
                    f"{api_url}/auth/google/session",
                    json=simple_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                print(f"   Simple session creation: {simple_response.status_code}")
                
                if simple_response.status_code == 200:
                    # Test the simple token
                    simple_auth_response = requests.get(
                        f"{api_url}/auth/me",
                        headers={'Authorization': f'Bearer {simple_token}'}
                    )
                    
                    print(f"   Simple token auth: {simple_auth_response.status_code}")
                    print(f"   Simple token response: {simple_auth_response.text}")
                    
                    if simple_auth_response.status_code == 200:
                        print("✅ Simple token works - issue might be with complex token format")
                    else:
                        print("❌ Simple token also fails - issue is deeper")
        else:
            print("❌ Session creation failed")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during investigation: {str(e)}")
    
    print("\n" + "=" * 60)
    print("INVESTIGATION SUMMARY:")
    print("- Check if session tokens are being stored correctly in user_sessions collection")
    print("- Check if get_current_user_enhanced is properly querying session tokens")
    print("- Check if there are any timing issues with session storage/retrieval")
    print("- Check if session token format affects storage/retrieval")

if __name__ == "__main__":
    investigate_session_token_issue()