"""
Device detection and management utilities
"""

import torch


def detect_device(device: str = None) -> str:
    """
    Smart device detection with fallback

    Args:
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device:
        return device

    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return 'cuda'
        else:
            return 'cpu'
    except (AssertionError, RuntimeError):
        print("⚠️  CUDA not available, falling back to CPU")
        return 'cpu'


def print_device_info(device: str):
    """Print device information and tips"""
    print(f"🚀 Using device: {device}")

    if device == 'cpu':
        print("💡 Tip: Install CUDA-enabled PyTorch for 10-20x speedup")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    else:
        # Print GPU info if available
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
        except:
            pass


def move_model_to_device(model, device: str, model_name: str = "Model"):
    """
    Safely move model to device with error handling

    Args:
        model: PyTorch model
        device: Target device
        model_name: Name of model for logging

    Returns:
        Tuple of (model, actual_device)
    """
    try:
        model = model.to(device)
        return model, device
    except (AssertionError, RuntimeError) as e:
        print(f"⚠️  Could not move {model_name} to {device}: {e}")
        print(f"   Falling back to CPU...")
        model = model.to('cpu')
        return model, 'cpu'