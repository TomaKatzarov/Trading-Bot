import os
import sys
import subprocess
from pathlib import Path

# Set environment variables to fix PyTorch + Streamlit conflicts
os.environ["STREAMLIT_WATCH_MODULES"] = ""
os.environ["STREAMLIT_SERVER_WATCH_DIRS"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def create_required_directories():
    """Create any required directories for the dashboard."""
    project_root = Path(__file__).parent.parent.absolute()
    
    # Create logs directory
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    # Create analysis/reports directory for plots
    reports_dir = project_root / 'analysis' / 'reports'
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Created/verified required directories")

def check_streamlit():
    """Checks if Streamlit is installed and installs it if missing."""
    try:
        import streamlit
        return True
    except ImportError:
        print("Streamlit not found. Installing streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            return True
        except Exception as e:
            print(f"Error installing streamlit: {e}")
            return False

def check_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        "matplotlib": "matplotlib",
        "seaborn": "seaborn", 
        "pandas": "pandas",
        "numpy": "numpy"
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            return False
    
    return True

def main():
    """Launches the model dashboard using streamlit."""
    print("Model Dashboard Launcher")
    print("=======================")
    
    # Create required directories
    create_required_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("Failed to install required dependencies.")
        return 1
    
    # Check if streamlit is installed
    if not check_streamlit():
        print("Failed to ensure streamlit is installed. Please install manually:")
        print("pip install streamlit")
        return 1
    
    # Get dashboard path
    dashboard_path = os.path.join(os.path.dirname(__file__), "model_dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        return 1
    
    print(f"Launching dashboard from: {dashboard_path}")
    print("Please wait while the dashboard starts...")
    
    # Launch streamlit with browser open flag and expanded environment variables
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, 
           "--server.headless=false",  # Open browser automatically
           "--server.runOnSave=true",  # Auto-reload on file save
           "--theme.base=light"]       # Light theme for better readability
    
    env = os.environ.copy()
    env["STREAMLIT_WATCH_MODULES"] = ""
    env["STREAMLIT_SERVER_WATCH_DIRS"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    try:
        subprocess.run(cmd, env=env)
        return 0
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        print("\nTo manually start the dashboard, run:")
        print(f"streamlit run {dashboard_path}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
