"""Initialize development environment."""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)

def create_virtual_env():
    """Create and activate virtual environment."""
    if not Path("venv").exists():
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Print activation instructions
    if os.name == "nt":  # Windows
        print("\nActivate virtual environment with:")
        print("    .\\venv\\Scripts\\activate")
    else:  # Unix
        print("\nActivate virtual environment with:")
        print("    source venv/bin/activate")

def install_dependencies():
    """Install Python dependencies."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=True)

def setup_frontend():
    """Install frontend dependencies."""
    frontend_dir = Path("src/frontend")
    if frontend_dir.exists():
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)

def download_model():
    """Download YOLOv8 weights."""
    subprocess.run([sys.executable, "scripts/download_weights.py"], check=True)

def main():
    """Run initialization steps."""
    try:
        print("Initializing development environment...")
        
        check_python_version()
        print("✓ Python version check passed")
        
        create_virtual_env()
        print("✓ Virtual environment created")
        
        install_dependencies()
        print("✓ Python dependencies installed")
        
        setup_frontend()
        print("✓ Frontend dependencies installed")
        
        download_model()
        print("✓ Model weights downloaded")
        
        print("\nDevelopment environment setup complete!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during initialization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()