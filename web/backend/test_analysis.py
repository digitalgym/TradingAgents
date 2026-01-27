"""
Integration tests for the analysis API endpoint.

Run with: python -m pytest test_analysis.py -v
Or directly: python test_analysis.py
"""
import os
import sys
import time
import json
import requests

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

BASE_URL = "http://127.0.0.1:8000"


def test_health_check():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("[OK] Health check passed")


def test_analysis_start():
    """Test starting an analysis task."""
    response = requests.post(
        f"{BASE_URL}/api/analysis/run",
        json={"symbol": "XAUUSD", "timeframe": "H1", "use_smc": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "started"
    print(f"[OK] Analysis started with task_id: {data['task_id']}")
    return data["task_id"]


def test_analysis_status(task_id: str):
    """Test getting analysis status."""
    response = requests.get(f"{BASE_URL}/api/analysis/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["running", "completed", "error"]
    print(f"[OK] Analysis status: {data['status']} ({data.get('progress', 0)}%)")
    return data


def test_analysis_completion(task_id: str, timeout: int = 300):
    """Test that analysis completes successfully within timeout."""
    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        response = requests.get(f"{BASE_URL}/api/analysis/status/{task_id}")
        data = response.json()

        if data["status"] != last_status:
            print(f"  Status: {data['status']} - {data.get('current_step_title', '')} ({data.get('progress', 0)}%)")
            last_status = data["status"]

        if data["status"] == "completed":
            print("[OK] Analysis completed successfully")
            assert "decision" in data
            print(f"  Decision: {json.dumps(data['decision'], indent=2)[:500]}...")
            return data
        elif data["status"] == "error":
            print(f"[FAIL] Analysis failed: {data.get('error', 'Unknown error')}")
            if "traceback" in data:
                print(f"  Traceback: {data['traceback'][:500]}...")
            return data

        time.sleep(5)

    print(f"[FAIL] Analysis timed out after {timeout} seconds")
    return None


def test_analysis_not_found():
    """Test getting status of non-existent task."""
    response = requests.get(f"{BASE_URL}/api/analysis/status/nonexistent_task_123")
    assert response.status_code == 404
    print("[OK] Not found error handled correctly")


def test_market_regime():
    """Test the market regime endpoint."""
    response = requests.get(f"{BASE_URL}/api/regime/XAUUSD?timeframe=H1")
    assert response.status_code == 200
    data = response.json()
    assert "regime" in data or "error" in data
    if "regime" in data:
        print(f"[OK] Market regime: {data['regime']} (volatility: {data.get('volatility', 'N/A')})")
    else:
        print(f"! Market regime error: {data.get('error', 'Unknown')}")


def run_full_integration_test():
    """Run a full integration test of the analysis pipeline."""
    print("\n" + "="*60)
    print("TradingAgents Analysis API Integration Test")
    print("="*60 + "\n")

    # Test health check
    test_health_check()

    # Test market regime
    test_market_regime()

    # Test 404 handling
    test_analysis_not_found()

    # Start and monitor analysis
    print("\n--- Starting Full Analysis Test ---")
    task_id = test_analysis_start()

    print("\nMonitoring analysis progress...")
    result = test_analysis_completion(task_id, timeout=300)

    if result and result["status"] == "completed":
        print("\n" + "="*60)
        print("[OK] INTEGRATION TEST PASSED")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("[FAIL] INTEGRATION TEST FAILED")
        print("="*60)
        return False


def run_quick_test():
    """Run a quick test without waiting for full analysis."""
    print("\n" + "="*60)
    print("TradingAgents Analysis API Quick Test")
    print("="*60 + "\n")

    test_health_check()
    test_market_regime()
    test_analysis_not_found()

    task_id = test_analysis_start()
    time.sleep(5)
    test_analysis_status(task_id)

    print("\n[OK] Quick test passed (analysis still running)")
    print(f"  Monitor with: curl {BASE_URL}/api/analysis/status/{task_id}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test the analysis API")
    parser.add_argument("--quick", action="store_true", help="Run quick test without waiting for completion")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for full test")
    args = parser.parse_args()

    try:
        # Check if server is running
        requests.get(f"{BASE_URL}/api/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server at {BASE_URL}")
        print("Make sure the backend server is running: python main.py")
        sys.exit(1)

    if args.quick:
        run_quick_test()
    else:
        success = run_full_integration_test()
        sys.exit(0 if success else 1)
