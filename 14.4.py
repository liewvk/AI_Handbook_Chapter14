import sys
import importlib.util

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
for package in ["numpy", "tensorflow", "torch"]:
    found = importlib.util.find_spec(package) is not None
    print(f"{package}: {'installed' if found else 'not installed'}")

