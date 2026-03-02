import os
import requests
from tqdm import tqdm
from pathlib import Path

# Configuration parameters
BASE_URL = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/year%3D2025/month%3D09/"
LOCAL_DIR = "./data"
# Set timeout duration (seconds)
TIMEOUT = 30
# Number of retry attempts for failed downloads
RETRY_TIMES = 3

# Optional: Specify target files to download directly (uncomment and fill in filenames)
TARGET_FILES = ["train-00000-of-00062.parquet", "train-00001-of-00062.parquet"]
# TARGET_FILES = None

def create_directory(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ready: {os.path.abspath(dir_path)}")

def get_file_list():
    """Get list of files from the target Hugging Face dataset directory"""
    # Hugging Face API endpoint to retrieve directory contents
    api_url = "https://huggingface.co/api/datasets/Lichess/standard-chess-games/tree/main/data/year%3D2025/month%3D09"
    
    try:
        response = requests.get(api_url, timeout=TIMEOUT)
        response.raise_for_status()  # Raise HTTP errors
        data = response.json()
        
        # Filter to get only files (exclude subdirectories)
        files = [item['path'].split('/')[-1] for item in data if item['type'] == 'file']
        if not files:
            print("⚠️ No files found")
            return []
        
        print(f"📄 Found {len(files)} files:")
        for idx, file in enumerate(files, 1):
            print(f"   {idx}. {file}")
        return files
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to retrieve file list: {e}")
        return []

def select_files(file_list):
    """Interactively select files to download (max 2 files)"""
    if TARGET_FILES:
        # Use predefined target files
        selected_files = [f for f in TARGET_FILES if f in file_list]
        missing_files = [f for f in TARGET_FILES if f not in file_list]
        if missing_files:
            print(f"⚠️ The following files do not exist: {missing_files}")
        if not selected_files:
            print("🚫 No valid predefined files")
            return []
        return selected_files[:2]  # Limit to maximum 2 files
    
    # Interactive selection
    print("\nPlease select files to download (enter numbers separated by commas, max 2 files):")
    while True:
        user_input = input("Enter numbers (e.g., 1,3): ").strip()
        if not user_input:
            print("🚫 Input cannot be empty, please try again")
            continue
        
        try:
            # Parse user input indices
            indices = [int(i.strip()) for i in user_input.split(',')]
            # Remove duplicates and limit to 2 files
            indices = list(set(indices))[:2]
            
            # Validate index range
            valid_indices = [i for i in indices if 1 <= i <= len(file_list)]
            if not valid_indices:
                print(f"🚫 Invalid numbers, please enter values between 1 and {len(file_list)}")
                continue
            
            # Get selected filenames
            selected_files = [file_list[i-1] for i in valid_indices]
            print(f"\n✅ Files you selected to download: {selected_files}")
            return selected_files
        
        except ValueError:
            print("🚫 Invalid input format, please enter numbers separated by commas")

def download_file(file_name, retry=0):
    """Download a single file with retry mechanism"""
    file_url = f"{BASE_URL}{file_name}"
    local_file_path = os.path.join(LOCAL_DIR, file_name)
    
    # Skip download if file already exists
    if os.path.exists(local_file_path):
        print(f"⏭️ {file_name} already exists, skipping download")
        return True
    
    try:
        # Send GET request with streaming to handle large files
        response = requests.get(file_url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()
        
        # Get total file size from response headers
        total_size = int(response.headers.get('content-length', 0))
        
        # Start downloading with progress bar
        with open(local_file_path, 'wb') as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        print(f"✅ {file_name} downloaded successfully")
        return True
    
    except requests.exceptions.RequestException as e:
        if retry < RETRY_TIMES:
            print(f"❌ Failed to download {file_name} (retry {retry+1}/{RETRY_TIMES}): {e}")
            return download_file(file_name, retry + 1)
        else:
            print(f"❌ Final failure to download {file_name}: {e}")
            # Remove incomplete file
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return False

def main():
    """Main function to orchestrate the download process"""
    print("🚀 Starting Lichess chess data download (support selective download)...")
    
    # 1. Create local directory
    create_directory(LOCAL_DIR)
    
    # 2. Get list of available files
    files = get_file_list()
    if not files:
        print("🚫 No downloadable files found, exiting program")
        return
    
    # 3. Select files to download (max 2)
    selected_files = select_files(files)
    if not selected_files:
        print("🚫 No files selected for download, exiting program")
        return
    
    # 4. Download selected files
    success_count = 0
    for file in selected_files:
        if download_file(file):
            success_count += 1
    
    # 5. Print download summary
    print("\n" + "="*50)
    print(f"📊 Download completed! Success: {success_count}/{len(selected_files)}")
    print(f"📁 Data saved to: {os.path.abspath(LOCAL_DIR)}")
    print("="*50)

if __name__ == "__main__":
    # Install dependencies if missing (first run)
    try:
        import tqdm
    except ImportError:
        print("⚠️ Required dependencies missing, installing automatically...")
        os.system("pip install requests tqdm")
        import tqdm
    
    main()