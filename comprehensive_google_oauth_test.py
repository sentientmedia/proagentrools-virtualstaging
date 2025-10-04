#!/usr/bin/env python3

import requests
import json
import sys
import uuid

def comprehensive_google_oauth_test():
    """Comprehensive test of Google OAuth integration addressing the review request"""
    
    base_url = "https://aihomedesign.preview.emergentagent.com"
    api_url = f"{base_url}/api"
    
    print("🔍 COMPREHENSIVE GOOGLE OAUTH INTEGRATION TEST")
    print("Addressing the specific issue from the review request")
    print("=" * 80)
    
    tests_run = 0
    tests_passed = 0
    issues_found = []
    
    def run_test(name, test_func):
        nonlocal tests_run, tests_passed
        tests_run += 1
        print(f"\n🔍 {name}...")
        try:
            result = test_func()
            if result['success']:
                tests_passed += 1
                print(f"✅ PASSED: {name}")
                if result.get('notes'):
                    print(f"   Notes: {result['notes']}")
                return True
            else:
                print(f"❌ FAILED: {name}")
                print(f"   Reason: {result.get('reason', 'Unknown')}")
                issues_found.append(f"{name}: {result.get('reason', 'Unknown')}")
                return False
        except Exception as e:
            print(f"❌ ERROR in {name}: {str(e)}")
            issues_found.append(f"{name}: Exception - {str(e)}")
            return False
    
    # Test 1: Fresh user creation with Google OAuth
    def test_fresh_user_creation():
        unique_id = uuid.uuid4().hex[:8]
        mock_data = {
            "user_data": {
                "id": f"google_fresh_{unique_id}",
                "email": f"fresh.oauth.{unique_id}@gmail.com", 
                "name": "Fresh OAuth User",
                "picture": "https://via.placeholder.com/150"
            },
            "session_token": f"fresh_session_{unique_id}"
        }
        
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=mock_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            return {'success': False, 'reason': f'Status {response.status_code}: {response.text}'}
        
        data = response.json()
        if not data.get('success'):
            return {'success': False, 'reason': f'Success flag false: {data}'}
        
        user = data.get('user', {})
        if user.get('credits') != 100:
            return {'success': False, 'reason': f'Expected 100 credits for new user, got {user.get("credits")}'}
        
        if user.get('subscription_status') != 'free':
            return {'success': False, 'reason': f'Expected free subscription, got {user.get("subscription_status")}'}
        
        # Store for next test
        test_fresh_user_creation.session_token = mock_data["session_token"]
        test_fresh_user_creation.email = mock_data["user_data"]["email"]
        
        return {
            'success': True, 
            'notes': f'New user created with 100 credits, email: {user.get("email")}'
        }
    
    # Test 2: Session token authentication immediately after creation
    def test_immediate_session_auth():
        if not hasattr(test_fresh_user_creation, 'session_token'):
            return {'success': False, 'reason': 'No session token from previous test'}
        
        session_token = test_fresh_user_creation.session_token
        
        response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if response.status_code != 200:
            return {'success': False, 'reason': f'Auth failed with status {response.status_code}: {response.text}'}
        
        data = response.json()
        expected_email = test_fresh_user_creation.email
        
        if data.get('email') != expected_email:
            return {'success': False, 'reason': f'Email mismatch: expected {expected_email}, got {data.get("email")}'}
        
        return {
            'success': True,
            'notes': f'Session token works immediately after creation for {data.get("email")}'
        }
    
    # Test 3: User sessions collection functionality
    def test_user_sessions_collection():
        if not hasattr(test_fresh_user_creation, 'session_token'):
            return {'success': False, 'reason': 'No session token from previous test'}
        
        session_token = test_fresh_user_creation.session_token
        
        # Test that session works for protected endpoints
        response = requests.get(
            f"{api_url}/auth/credits",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if response.status_code != 200:
            return {'success': False, 'reason': f'Credits endpoint failed: {response.text}'}
        
        data = response.json()
        if 'credits' not in data or 'subscription_status' not in data:
            return {'success': False, 'reason': f'Missing fields in credits response: {data}'}
        
        return {
            'success': True,
            'notes': f'User sessions collection working - credits: {data.get("credits")}'
        }
    
    # Test 4: Existing user session update (core issue from review)
    def test_existing_user_session_update():
        if not hasattr(test_fresh_user_creation, 'email'):
            return {'success': False, 'reason': 'No email from previous test'}
        
        # Use same email but new session token
        existing_email = test_fresh_user_creation.email
        new_session_token = f"updated_session_{uuid.uuid4().hex[:8]}"
        
        update_data = {
            "user_data": {
                "id": f"google_updated_{uuid.uuid4().hex[:8]}",
                "email": existing_email,  # Same email
                "name": "Updated OAuth User",
                "picture": "https://via.placeholder.com/200"
            },
            "session_token": new_session_token
        }
        
        response = requests.post(
            f"{api_url}/auth/google/session",
            json=update_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            return {'success': False, 'reason': f'Session update failed: {response.text}'}
        
        # Test new session token
        auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {new_session_token}'}
        )
        
        if auth_response.status_code != 200:
            return {'success': False, 'reason': f'New session auth failed: {auth_response.text}'}
        
        auth_data = auth_response.json()
        if auth_data.get('email') != existing_email:
            return {'success': False, 'reason': f'Email mismatch after update: {auth_data.get("email")}'}
        
        # Store new token for next tests
        test_existing_user_session_update.session_token = new_session_token
        
        return {
            'success': True,
            'notes': f'Existing user session updated successfully for {existing_email}'
        }
    
    # Test 5: Credit deduction with session token
    def test_credit_deduction_with_session():
        if not hasattr(test_existing_user_session_update, 'session_token'):
            return {'success': False, 'reason': 'No session token from previous test'}
        
        session_token = test_existing_user_session_update.session_token
        
        # Get initial credits
        credits_response = requests.get(
            f"{api_url}/auth/credits",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if credits_response.status_code != 200:
            return {'success': False, 'reason': f'Could not get initial credits: {credits_response.text}'}
        
        initial_credits = credits_response.json().get('credits', 0)
        
        # Use interior design tool to deduct credits
        try:
            files = {'file': ('test_credit_deduction.jpg', b'fake_image_data', 'image/jpeg')}
            data = {
                'room_type': 'bedroom',
                'designer': 'adrian_mercer',
                'color_scheme': 'nomad_prism'
            }
            
            design_response = requests.post(
                f"{api_url}/interior-design/process",
                data=data,
                files=files,
                headers={'Authorization': f'Bearer {session_token}'}
            )
            
            if design_response.status_code != 200:
                return {'success': False, 'reason': f'Interior design failed: {design_response.text}'}
            
            design_data = design_response.json()
            credits_used = design_data.get('credits_used', 0)
            remaining_credits = design_data.get('remaining_credits', 0)
            
            if credits_used <= 0:
                return {'success': False, 'reason': f'No credits deducted: {credits_used}'}
            
            if remaining_credits != (initial_credits - credits_used):
                return {'success': False, 'reason': f'Credit calculation error: {initial_credits} - {credits_used} != {remaining_credits}'}
            
            return {
                'success': True,
                'notes': f'Credits deducted correctly: {initial_credits} -> {remaining_credits} (used {credits_used})'
            }
            
        except Exception as e:
            return {'success': False, 'reason': f'Exception during credit deduction test: {str(e)}'}
    
    # Test 6: Session expiry handling
    def test_session_expiry_handling():
        # Test with a fake expired token
        fake_expired_token = f"expired_token_{uuid.uuid4().hex}"
        
        response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {fake_expired_token}'}
        )
        
        if response.status_code == 200:
            return {'success': False, 'reason': 'Fake expired token was accepted (should be rejected)'}
        
        if response.status_code != 401:
            return {'success': False, 'reason': f'Expected 401 for expired token, got {response.status_code}'}
        
        return {
            'success': True,
            'notes': 'Invalid/expired session tokens correctly rejected with 401'
        }
    
    # Test 7: Logout and session cleanup
    def test_logout_and_cleanup():
        if not hasattr(test_existing_user_session_update, 'session_token'):
            return {'success': False, 'reason': 'No session token from previous test'}
        
        session_token = test_existing_user_session_update.session_token
        
        # Test logout
        logout_response = requests.post(
            f"{api_url}/auth/logout",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if logout_response.status_code != 200:
            return {'success': False, 'reason': f'Logout failed: {logout_response.text}'}
        
        logout_data = logout_response.json()
        if not logout_data.get('success'):
            return {'success': False, 'reason': f'Logout success flag false: {logout_data}'}
        
        # Test that session is now invalid
        auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {session_token}'}
        )
        
        if auth_response.status_code == 200:
            return {'success': False, 'reason': 'Session token still works after logout'}
        
        return {
            'success': True,
            'notes': 'Logout successfully invalidates session tokens'
        }
    
    # Test 8: Mixed authentication (JWT + Session tokens)
    def test_mixed_authentication():
        # Create a JWT user
        jwt_user_data = {
            "email": f"jwt.user.{uuid.uuid4().hex[:8]}@example.com",
            "password": "testpassword123",
            "full_name": "JWT Test User"
        }
        
        jwt_response = requests.post(
            f"{api_url}/auth/register",
            json=jwt_user_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if jwt_response.status_code != 200:
            return {'success': False, 'reason': f'JWT user creation failed: {jwt_response.text}'}
        
        jwt_data = jwt_response.json()
        jwt_token = jwt_data.get('access_token')
        
        # Test JWT authentication
        jwt_auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {jwt_token}'}
        )
        
        if jwt_auth_response.status_code != 200:
            return {'success': False, 'reason': f'JWT authentication failed: {jwt_auth_response.text}'}
        
        # Create a Google OAuth user
        oauth_data = {
            "user_data": {
                "id": f"google_mixed_{uuid.uuid4().hex[:8]}",
                "email": f"mixed.oauth.{uuid.uuid4().hex[:8]}@gmail.com",
                "name": "Mixed Auth OAuth User",
                "picture": "https://via.placeholder.com/150"
            },
            "session_token": f"mixed_session_{uuid.uuid4().hex[:8]}"
        }
        
        oauth_response = requests.post(
            f"{api_url}/auth/google/session",
            json=oauth_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if oauth_response.status_code != 200:
            return {'success': False, 'reason': f'OAuth user creation failed: {oauth_response.text}'}
        
        # Test OAuth authentication
        oauth_auth_response = requests.get(
            f"{api_url}/auth/me",
            headers={'Authorization': f'Bearer {oauth_data["session_token"]}'}
        )
        
        if oauth_auth_response.status_code != 200:
            return {'success': False, 'reason': f'OAuth authentication failed: {oauth_auth_response.text}'}
        
        return {
            'success': True,
            'notes': 'Both JWT and Google OAuth session authentication work simultaneously'
        }
    
    # Run all tests
    run_test("Fresh user creation with Google OAuth", test_fresh_user_creation)
    run_test("Session token authentication immediately after creation", test_immediate_session_auth)
    run_test("User sessions collection functionality", test_user_sessions_collection)
    run_test("Existing user session update (core issue)", test_existing_user_session_update)
    run_test("Credit deduction with session token", test_credit_deduction_with_session)
    run_test("Session expiry handling", test_session_expiry_handling)
    run_test("Logout and session cleanup", test_logout_and_cleanup)
    run_test("Mixed authentication (JWT + Session tokens)", test_mixed_authentication)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"📊 COMPREHENSIVE GOOGLE OAUTH TEST RESULTS:")
    print(f"   Tests Run: {tests_run}")
    print(f"   Tests Passed: {tests_passed}")
    print(f"   Success Rate: {(tests_passed/tests_run)*100:.1f}%")
    
    if issues_found:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"   - {issue}")
    
    if tests_passed == tests_run:
        print("\n🎉 ALL GOOGLE OAUTH TESTS PASSED!")
        print("✅ Google OAuth integration is working correctly")
        print("✅ Session creation, authentication, and cleanup all functional")
        print("✅ Credit deduction works with session tokens")
        print("✅ Mixed authentication (JWT + OAuth) works")
        return 0
    else:
        print(f"\n⚠️  {tests_run - tests_passed} tests failed")
        print("❌ Some Google OAuth functionality issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(comprehensive_google_oauth_test())