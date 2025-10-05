#!/usr/bin/env python3
"""
Focused testing for Phase 2 (MCP Mega-Agent) and Phase 3 (Watermarking System)
"""

import sys
import os
sys.path.append('/app')

from backend_test import ProAgentToolsAPITester

def test_phase_2_and_3():
    """Test Phase 2 and Phase 3 features specifically"""
    print("🚀 Testing Phase 2 (MCP Mega-Agent) and Phase 3 (Watermarking System)")
    print("=" * 80)
    
    tester = ProAgentToolsAPITester()
    
    # First ensure we have authentication
    print("\n🔐 Setting up authentication...")
    tester.test_user_registration()
    
    # ========== PHASE 2: MCP MEGA-AGENT TESTING ==========
    print("\n🤖 PHASE 2: Testing MCP Mega-Agent System...")
    
    phase2_tests = [
        ("Mega-Agent AI Tools Processing", tester.test_mega_agent_ai_tools_processing),
        ("Mega-Agent Insufficient Credits", tester.test_mega_agent_insufficient_credits),
        ("Get AI Results Endpoint", tester.test_get_ai_results_endpoint),
        ("Mega-Agent Tool Categorization", tester.test_mega_agent_tool_categorization),
        ("Mega-Agent Unified Summary", tester.test_mega_agent_unified_summary)
    ]
    
    phase2_passed = 0
    for test_name, test_func in phase2_tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                phase2_passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {str(e)}")
    
    # ========== PHASE 3: WATERMARKING SYSTEM TESTING ==========
    print("\n🎨 PHASE 3: Testing Watermarking System...")
    
    phase3_tests = [
        ("Upload Agent Logo", tester.test_upload_agent_logo),
        ("Upload Invalid Logo File", tester.test_upload_invalid_logo_file),
        ("Serve Agent Logo", tester.test_serve_agent_logo),
        ("Serve Non-existent Logo", tester.test_serve_nonexistent_logo),
        ("Get Branding Settings", tester.test_get_branding_settings),
        ("Update Branding Settings", tester.test_update_branding_settings),
        ("Invalid Branding Settings", tester.test_invalid_branding_settings),
        ("Watermarking Integration", tester.test_watermarking_integration),
        ("Watermark Storage Directories", tester.test_watermark_storage_directories),
        ("Watermark File Validation", tester.test_watermark_file_validation)
    ]
    
    phase3_passed = 0
    for test_name, test_func in phase3_tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                phase3_passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {str(e)}")
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 PHASE 2 & 3 TEST RESULTS:")
    print(f"   Phase 2 (MCP Mega-Agent): {phase2_passed}/{len(phase2_tests)} tests passed")
    print(f"   Phase 3 (Watermarking): {phase3_passed}/{len(phase3_tests)} tests passed")
    print(f"   Overall: {phase2_passed + phase3_passed}/{len(phase2_tests) + len(phase3_tests)} tests passed")
    
    total_tests = len(phase2_tests) + len(phase3_tests)
    total_passed = phase2_passed + phase3_passed
    success_rate = (total_passed / total_tests) * 100
    
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Phase 2 & 3 testing successful!")
        return 0
    else:
        print("⚠️ Some Phase 2 & 3 tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(test_phase_2_and_3())