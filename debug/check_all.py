import subprocess
import sys
import os

# List of debug scripts to check before training.
# Edit/add/remove as needed to match your folder!
DEBUG_SCRIPTS = [
    "device_check.py",
    "oom_check.py",
    "loader_check.py",
    "loss_check.py",
    "grad_check.py",
    "sample_output.py"
]

def print_header(msg):
    print("\n" + "="*40)
    print(f"{msg}")
    print("="*40)

def run_script(script):
    print_header(f"Running: {script}")
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode == 0:
        print(f"✅ {script}: PASS\n")
    else:
        print(f"❌ {script}: FAIL\n")
        print("Error:", result.stderr)
    return result.returncode == 0

def main():
    all_passed = True
    print_header("APT Master Debug - Pre-Training Smoke Test")
    for script in DEBUG_SCRIPTS:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if not os.path.exists(script_path):
            print(f"⚠️  {script} not found! Skipping.")
            continue
        passed = run_script(script)
        if not passed:
            all_passed = False

    print_header("SUMMARY")
    if all_passed:
        print("🚀 ALL CHECKS PASSED — SAFE TO START TRAINING!")
        sys.exit(0)
    else:
        print("❌ ONE OR MORE CHECKS FAILED — FIX BEFORE STARTING TRAINING!")
        sys.exit(1)

if __name__ == "__main__":
    main()
