#!/usr/bin/env python3
"""
Pre-flight Check Script for main.py
Validates environment and dependencies before running TTS synthesis
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10"""
    print("Checking Python version...", end=" ")
    if sys.version_info < (3, 10):
        print(f"✗ FAIL")
        print(f"  Current: {sys.version_info.major}.{sys.version_info.minor}")
        print(f"  Required: 3.10+")
        return False
    print(f"✓ OK ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
    return True


def check_dependencies():
    """Check required packages are installed"""
    print("Checking dependencies...")

    required = ["httpx", "dotenv"]
    all_ok = True

    for package in required:
        print(f"  {package}...", end=" ")
        try:
            if package == "dotenv":
                import dotenv
            else:
                __import__(package)
            print("✓ OK")
        except ImportError:
            print("✗ MISSING")
            all_ok = False

    return all_ok


def check_env_file():
    """Check if .env file exists"""
    print("Checking .env file...", end=" ")
    env_path = Path(".env")

    if not env_path.exists():
        print("✗ NOT FOUND")
        print("  Create .env file with: CARTESIA_API_KEY=your_key")
        return False

    print("✓ EXISTS")
    return True


def check_api_key():
    """Check if API key is set"""
    print("Checking API key...", end=" ")

    try:
        from dotenv import load_dotenv
        import os

        load_dotenv()
        api_key = os.getenv("CARTESIA_API_KEY")

        if not api_key:
            print("✗ NOT SET")
            print("  Set CARTESIA_API_KEY in .env file")
            return False

        if api_key == "your_api_key_here" or len(api_key) < 10:
            print("✗ INVALID")
            print("  Replace placeholder with actual API key")
            return False

        print(f"✓ OK ({api_key[:12]}...)")
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def check_file_permissions():
    """Check main.py exists and is executable"""
    print("Checking main.py...", end=" ")

    main_py = Path("main.py")

    if not main_py.exists():
        print("✗ NOT FOUND")
        return False

    print("✓ EXISTS")
    return True


def main():
    """Run all checks"""
    print("=" * 70)
    print("Pre-flight Check for main.py")
    print("=" * 70)
    print()

    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment file", check_env_file),
        ("API key", check_api_key),
        ("Script file", check_file_permissions),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"✗ ERROR in {name}: {e}")
            results.append(False)
        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print()
        print("✓ All checks passed! Ready to run:")
        print("  python main.py")
        return 0
    else:
        print()
        print("✗ Some checks failed. Fix issues above before running main.py")
        print()
        print("Quick fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Create .env file: cp .env.template .env")
        print("  3. Add API key to .env: CARTESIA_API_KEY=your_key_here")
        return 1


if __name__ == "__main__":
    sys.exit(main())
