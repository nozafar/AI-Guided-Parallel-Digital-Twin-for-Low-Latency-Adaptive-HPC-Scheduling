import os
import urllib.request
import zipfile
import argparse

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    
    class DownloadProgressBar:
        def __init__(self):
            self.pbar = None
            
        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                print(f"Total file size: {total_size / (1024 * 1024):.2f} MB")
                self.pbar = True
            
            downloaded = block_num * block_size
            if downloaded % (100 * 1024 * 1024) == 0 or downloaded >= total_size:
                print(f"Downloaded: {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB")

    urllib.request.urlretrieve(url, dest_path, reporthook=DownloadProgressBar())
    print("Download complete.")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def main():
    parser = argparse.ArgumentParser(description="Download Alibaba GPU Cluster Trace 2026")
    parser.add_argument("--full", action="store_true", help="Download all files including the 351GB pod hourly data.")
    args = parser.parse_args()
    
    base_url = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/"
    data_dir = "data/alibaba_raw"
    os.makedirs(data_dir, exist_ok=True)
    
    files = [
        ("asi_opensource_job_execution_summary.zip", 1.18),
        ("asi_opensource_network_hourly.zip", 0.2)
    ]
    
    if args.full:
        files.extend([
            ("asi_opensource_server_hourly.zip", 3.0),
            ("asi_opensource_pod_hourly.zip", 351.8)
        ])
        
    print(f"Starting download of {len(files)} files...")
    
    for filename, size_gb in files:
        url = base_url + filename
        zip_path = os.path.join(data_dir, filename)
        
        print(f"\n--- Processing {filename} (~{size_gb} GB) ---")
        if not os.path.exists(zip_path):
            download_file(url, zip_path)
        else:
            print(f"{filename} already exists. Skipping download.")
            
        extract_dir = os.path.join(data_dir, filename.replace('.zip', ''))
        if not os.path.exists(extract_dir):
            extract_zip(zip_path, data_dir)
        else:
            print(f"{filename} is already extracted.")
            
    print("\nAll downloads and extractions finished.")

if __name__ == "__main__":
    main()
