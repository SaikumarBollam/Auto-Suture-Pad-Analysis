"""Initialize Git repository and set up initial commit."""

import os
import subprocess
from pathlib import Path

def init_git():
    """Initialize Git repository and create initial commit."""
    try:
        # Initialize git repository if not already initialized
        if not Path(".git").exists():
            subprocess.run(["git", "init"], check=True)
            print("✓ Git repository initialized")
            
        # Create initial commit if no commits exist
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Stage all files
            subprocess.run(["git", "add", "."], check=True)
            
            # Create initial commit
            subprocess.run(
                ["git", "commit", "-m", "Initial commit: Project setup"],
                check=True
            )
            print("✓ Initial commit created")
        else:
            print("Repository already has commits")
            
    except subprocess.CalledProcessError as e:
        print(f"Error initializing Git repository: {e}")
        return False
        
    return True

if __name__ == "__main__":
    init_git()