import subprocess
import sys
import os

DEBUG_SCRIPTS = [
    "device_check.py",
    "oom_check.py",
    "loader_check.py",
    "loss_check.py",
    "grad_check.py",
    "sample_output.py",
    "debug_pipeline.py"   # <<-- add the new full pipeline debug here!
]

print("="*44)
print("MASTER DEBUG: RUNNING ALL CHECKS")
print("="*44)

def run_script(script):
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), script)])
    if result.returncode == 0:
        print(f"✅ {script} PASSED")
    else:
        print(f"❌ {script} FAILED (see output above)")
        sys.exit(result.returncode)

for script in DEBUG_SCRIPTS:
    run_script(script)

print("\n🚀 ALL CHECKS PASSED — SAFE TO START TRAINING!")
