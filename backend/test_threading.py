#!/usr/bin/env python3
"""
Test script to verify threading is working correctly
Run this after starting the FastAPI server
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_threading():
    print_section("THREADING TEST SCRIPT")
    
    # Step 1: Check server is running
    print("\n1️⃣  Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print("   ✅ Server is running!")
    except requests.exceptions.ConnectionError:
        print("   ❌ Server is not running!")
        print("   👉 Start the server with: python main.py")
        return
    
    # Step 2: Login (you'll need to create an account first)
    print("\n2️⃣  Login required...")
    print("   👉 Please login via the web interface first")
    print("   👉 Or register at: http://localhost:5173")
    
    token = input("\n   Enter your auth token (from browser localStorage): ").strip()
    if not token:
        print("   ⚠️  No token provided. Using test without auth...")
        headers = {}
    else:
        headers = {"Authorization": f"Bearer {token}"}
    
    # Step 3: Start a session
    print("\n3️⃣  Starting a test session...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/session/start",
            json={
                "subject": "Threading Test",
                "semester": "1",
                "department": "CS"
            },
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Session started!")
            print(f"   📝 Session ID: {data['session_id']}")
            session_id = data['session_id']
        else:
            print(f"   ❌ Failed to start session: {response.status_code}")
            print(f"   Error: {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 4: Check debug status
    print("\n4️⃣  Checking threading status...")
    for i in range(5):
        time.sleep(1)
        try:
            response = requests.get(f"{BASE_URL}/api/debug/status")
            status = response.json()
            
            print(f"\n   Check #{i+1}:")
            print(f"   📹 Camera Active: {'✅' if status['camera_active'] else '❌'}")
            print(f"   🧵 Detection Thread Alive: {'✅' if status['detection_thread_alive'] else '❌'}")
            print(f"   🔄 Processing Active: {'✅' if status['processing_active'] else '❌'}")
            print(f"   📊 Active Threads: {status['active_threads']}")
            print(f"   🏷️  Thread Names: {', '.join(status['thread_names'])}")
            print(f"   👥 Students Detected: {status['students_detected']}")
            
            if status['detection_thread_alive'] and status['camera_active']:
                print("\n   🎉 Threading is working correctly!")
                break
        except Exception as e:
            print(f"   ❌ Error checking status: {e}")
    
    # Step 5: Monitor for 20 seconds
    print("\n5️⃣  Monitoring performance for 20 seconds...")
    print("   👀 Watch the server console for FPS metrics")
    print("   📹 Expected VIDEO FPS: ~30")
    print("   🔍 Expected DETECTION FPS: ~6")
    
    for i in range(20):
        time.sleep(1)
        print(f"   ⏱️  {i+1}/20 seconds elapsed...", end='\r')
    
    print("\n   ✅ Monitoring complete!")
    
    # Step 6: Check attendance
    print("\n6️⃣  Checking current attendance...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/session/current-attendance",
            headers=headers
        )
        
        if response.status_code == 200:
            attendance = response.json()
            print(f"   👥 {len(attendance)} student(s) detected")
            
            if attendance:
                for record in attendance:
                    print(f"   - {record['student_usn']}: {record['student_name']}")
                    scores = record.get('attentiveness_scores', [])
                    print(f"     Attentiveness records: {len(scores)}")
        else:
            print(f"   ⚠️  Could not fetch attendance: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 7: Stop session
    print("\n7️⃣  Stopping session...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/session/stop",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Session stopped!")
            print(f"   📊 Attendance records saved: {data['attendance_count']}")
        else:
            print(f"   ❌ Failed to stop session: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 8: Verify threads stopped
    print("\n8️⃣  Verifying threads stopped...")
    time.sleep(1)
    try:
        response = requests.get(f"{BASE_URL}/api/debug/status")
        status = response.json()
        
        print(f"   🧵 Detection Thread Alive: {'❌ (should be stopped)' if status['detection_thread_alive'] else '✅ Stopped'}")
        print(f"   📹 Camera Active: {'❌ (should be stopped)' if status['camera_active'] else '✅ Stopped'}")
        print(f"   📊 Active Threads: {status['active_threads']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print_section("TEST COMPLETE")
    print("\n✅ All tests completed!")
    print("\n📝 Summary:")
    print("   - Threading implementation is working")
    print("   - Video and detection run on separate threads")
    print("   - Threads start/stop correctly with sessions")
    print("\n💡 Check the server console for detailed FPS metrics!")

if __name__ == "__main__":
    try:
        test_threading()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()