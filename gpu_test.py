"""
GPU Diagnostic Script
"""
import torch

print("="*60)
print("GPU Diagnostic Information")
print("="*60)

# Check CUDA availability
print(f"\nCUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3:.2f} GB")
    
    # Test GPU
    print("\nTesting GPU...")
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    print(f"✓ GPU computation successful!")
    print(f"Result device: {z.device}")
else:
    print("\n❌ CUDA not available!")
    print("\nPossible issues:")
    print("1. PyTorch CPU-only version installed")
    print("2. CUDA drivers not installed")
    print("3. GPU not detected")
    
    print("\nTo fix:")
    print("Visit: https://pytorch.org/get-started/locally/")
    print("Select your configuration and reinstall PyTorch with CUDA support")

print("="*60)