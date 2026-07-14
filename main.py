import platform
import torch
import numpy as np


def main():
    print("=" * 50)
    print(f"{'Welcome to Glassbase':^50}")
    print("=" * 50)
    print(f"Python      : {platform.python_version()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA        : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"NumPy       : {np.__version__}")
    print(f"System      : {platform.system()} {platform.release()}")
    print(f"Device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
