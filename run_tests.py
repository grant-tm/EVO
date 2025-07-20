#!/usr/bin/env python3
"""
Test runner for the EVO trading system.

This script provides convenient ways to run different types of tests
for the EVO system.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run EVO tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "core", "config", "logging", "utils", "all"],
        default="unit",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        help="Specific pytest markers to run"
    )
    parser.add_argument(
        "--file",
        help="Run tests from specific file"
    )
    parser.add_argument(
        "--function",
        help="Run specific test function"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add markers based on test type
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "core":
        cmd.extend(["-m", "core"])
    elif args.type == "config":
        cmd.extend(["-m", "config"])
    elif args.type == "logging":
        cmd.extend(["-m", "logging"])
    elif args.type == "utils":
        cmd.extend(["-m", "utils"])
    elif args.type == "all":
        pass  # Run all tests
    
    # Add custom markers
    if args.markers:
        cmd.extend(["-m", " and ".join(args.markers)])
    
    # Add file filter
    if args.file:
        cmd.append(args.file)
    
    # Add function filter
    if args.function:
        cmd.extend(["-k", args.function])
    
    # Add verbose flag
    if args.verbose:
        cmd.append("-vv")
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=evo", "--cov-report=html", "--cov-report=term-missing"])
    
    # Exclude slow tests unless requested
    if not args.slow:
        cmd.extend(["-m", "not slow"])
    
    # Run the tests
    success = run_command(cmd, f"EVO {args.type} tests")
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 