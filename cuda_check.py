"""
CUDA Installation Check
"""
import torch
import sys

print("="*60)
print("PyTorch CUDA Status Check")
print("="*60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ CUDA is working!")
    print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("\n✅ You're good to go! No need to install CUDA separately.")
else:
    print(f"❌ CUDA is NOT available!")
    print(f"\nYour PyTorch version: {torch.__version__}")
    
    if 'cpu' in torch.__version__:
        print("❌ Problem: You installed CPU-only PyTorch")
        print("\n🔧 Solution: Reinstall PyTorch with CUDA support")
    else:
        print("❌ Problem: CUDA drivers may not be installed")
        print("\n🔧 Solution: Install NVIDIA CUDA drivers")

print("="*60)