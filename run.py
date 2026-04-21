import sys
import os

# Add the project directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main import main

if __name__ == "__main__":
    main()
