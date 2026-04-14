"""
Load Testing Script for FraudShield API.

Usage:
    python scripts/load_test.py --duration 60 --workers 10
    python scripts/load_test.py --spike
"""

import argparse
import random
import statistics
import time
import concurrent.futures
from typing import Tuple
import requests

API_URL = "http://localhost:8000"

# Sample transactions
NORMAL_TRANSACTION = {
    "transaction_amount": 2.0,
    "transaction_time": "14:30",
    "transaction_date": "2025-03-08",
    "transaction_type": "POS",
    "merchant_category": "Retail",
    "transaction_location": "Singapore",
    "customer_home_location": "Singapore",
    "distance_from_home": 5.0,
    "card_type": "Credit",
    "account_balance": 50.0,
    "daily_transaction_count": 2,
    "weekly_transaction_count": 8,
    "avg_transaction_amount": 2.5,
    "max_transaction_last_24h": 3.0,
    "is_international_transaction": False,
    "is_new_merchant": False,
    "failed_transaction_count": 0,
    "unusual_time_transaction": False,
    "previous_fraud_count": 0,
}

SUSPICIOUS_TRANSACTION = {
    "transaction_amount": 45.0,
    "transaction_time": "03:15",
    "transaction_date": "2025-03-08",
    "transaction_type": "Online",
    "merchant_category": "Electronics",
    "transaction_location": "Tokyo",
    "customer_home_location": "Lagos",
    "distance_from_home": 13000.0,
    "card_type": "Credit",
    "account_balance": 1.5,
    "daily_transaction_count": 20,
    "weekly_transaction_count": 60,
    "avg_transaction_amount": 1.0,
    "max_transaction_last_24h": 45.0,
    "is_international_transaction": True,
    "is_new_merchant": True,
    "failed_transaction_count": 4,
    "unusual_time_transaction": True,
    "previous_fraud_count": 1,
}


def random_transaction() -> dict:
    """Generate a random realistic transaction."""
    is_suspicious = random.random() < 0.1  # 10% suspicious
    base = SUSPICIOUS_TRANSACTION if is_suspicious else NORMAL_TRANSACTION
    tx = base.copy()
    tx["transaction_amount"] = round(random.uniform(0.5, 50.0), 1)
    tx["account_balance"] = round(random.uniform(1.0, 100.0), 1)
    return tx


def make_prediction() -> Tuple[bool, float, str]:
    """Make a single prediction request. Returns (success, latency_ms, risk_level)."""
    start = time.time()
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=random_transaction(),
            timeout=5,
        )
        latency_ms = (time.time() - start) * 1000
        if response.status_code == 200:
            return True, latency_ms, response.json().get("risk_level", "UNKNOWN")
        return False, latency_ms, "ERROR"
    except Exception:
        return False, (time.time() - start) * 1000, "ERROR"


def check_health() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


def print_statistics(total, successful, latencies, fraud_counts, duration):
    print("\n" + "=" * 60)
    print("Load Test Results")
    print("=" * 60)
    print(f"Total Requests:      {total}")
    print(f"Successful:          {successful}")
    print(f"Failed:              {total - successful}")
    if total > 0:
        print(f"Success Rate:        {successful / total * 100:.1f}%")
    print(f"Requests/Second:     {total / duration:.1f}")

    if latencies:
        sl = sorted(latencies)
        print(f"\nLatency (ms):")
        print(f"  Mean:   {statistics.mean(latencies):.1f}")
        print(f"  Median: {statistics.median(latencies):.1f}")
        print(f"  P95:    {sl[int(len(sl) * 0.95)]:.1f}")
        print(f"  P99:    {sl[min(int(len(sl) * 0.99), len(sl)-1)]:.1f}")

    if fraud_counts:
        total_preds = sum(fraud_counts.values())
        print(f"\nRisk Level Distribution:")
        for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = fraud_counts.get(level, 0)
            pct = count / total_preds * 100 if total_preds > 0 else 0
            print(f"  {level:<10}: {count:>5} ({pct:.1f}%)")
    print("=" * 60)


def run_load_test(duration=60, workers=10):
    print("=" * 60)
    print("FraudShield API Load Test")
    print(f"Duration: {duration}s | Workers: {workers}")
    print("=" * 60)

    if not check_health():
        print("ERROR: API not healthy. Start with: docker compose up -d")
        return

    total, successful = 0, 0
    latencies = []
    risk_counts = {}
    start_time = time.time()
    last_report = start_time

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while time.time() - start_time < duration:
            futures = [executor.submit(make_prediction) for _ in range(workers)]
            for future in concurrent.futures.as_completed(futures):
                success, latency, risk = future.result()
                total += 1
                if success:
                    successful += 1
                    risk_counts[risk] = risk_counts.get(risk, 0) + 1
                latencies.append(latency)

            if time.time() - last_report >= 10:
                elapsed = time.time() - start_time
                fraud = risk_counts.get("HIGH", 0) + risk_counts.get("CRITICAL", 0)
                print(f"  [{int(elapsed):3d}s] {total} reqs | {total/elapsed:.1f} rps | HIGH/CRITICAL: {fraud}")
                last_report = time.time()

            time.sleep(0.05)

    print_statistics(total, successful, latencies, risk_counts, duration)


def run_spike_test(normal_workers=5, spike_workers=50, spike_duration=15):
    print("=" * 60)
    print("Spike Test")
    print("=" * 60)

    if not check_health():
        print("ERROR: API not healthy.")
        return

    total, successful = 0, 0
    latencies = []

    def run_phase(workers, dur, label):
        nonlocal total, successful
        phase_start = time.time()
        print(f"\nPhase: {label} ({workers} workers, {dur}s)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            while time.time() - phase_start < dur:
                futures = [executor.submit(make_prediction) for _ in range(workers)]
                for f in concurrent.futures.as_completed(futures):
                    ok, lat, _ = f.result()
                    total += 1
                    if ok:
                        successful += 1
                    latencies.append(lat)
                time.sleep(0.1)

    run_phase(normal_workers, 30, "Normal load")
    run_phase(spike_workers, spike_duration, "🔴 SPIKE")
    run_phase(normal_workers, 30, "Recovery")

    print_statistics(total, successful, latencies, {}, 30 + spike_duration + 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--spike", action="store_true")
    args = parser.parse_args()

    if args.spike:
        run_spike_test(normal_workers=args.workers)
    else:
        run_load_test(args.duration, args.workers)


if __name__ == "__main__":
    main()