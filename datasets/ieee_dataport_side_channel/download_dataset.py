#!/usr/bin/env python3
"""
Downloader script for the IEEE DataPort "Hardware Trojan Power & EM Side-Channel Dataset".
DOI: 10.21227/9fwb-8978
Page: https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset

Usage:
  1. Login to https://ieee-dataport.org in your web browser (free account, no IEEE membership required).
  2. Open Developer Tools -> Application / Storage -> Cookies -> https://ieee-dataport.org
  3. Copy your session cookie value (starts with 'SSESS...').
  4. Run this script:
     python download_dataset.py --session-cookie "SSESS<your_cookie_value>"
     
Alternatively, you can manually download the 12 zip files from the page and place them in the 'data/' folder.
"""

import os
import sys
import argparse
import urllib.request
import re

DATASET_PAGE = "https://ieee-dataport.org/open-access/hardware-trojan-power-em-side-channel-dataset"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES = [
    "AES-T400_power_Temp25C.zip",
    "AES-T500_power_Temp25C.zip",
    "AES-T600_power_Temp25C.zip",
    "AES-T700_power_Temp25C.zip",
    "AES-T800_power_Temp25C.zip",
    "AES-T1000_power_Temp25C.zip",
    "AES-T1100_power_Temp25C.zip",
    "AES-T1300_power_Temp25C.zip",
    "AES-T1400_power_Temp25C.zip",
    "AES-T1600_power_Temp25C.zip",
    "AES-T1800_power_Temp25C.zip",
    "AES-T2000_power_Temp25C.zip",
]

def download_with_cookie(cookie_string):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Cookie": cookie_string
    }
    
    print(f"[*] Accessing dataset landing page with session cookie...")
    req = urllib.request.Request(DATASET_PAGE, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"[-] Failed to access dataset page: {e}")
        return

    # Find presigned S3 or direct download links
    download_links = re.findall(r'href=[\"\'](https://[^\"\']*(?:amazonaws\.com|download|files)[^\"\']*)[\"\']', html, re.I)
    print(f"[*] Found {len(download_links)} downloadable links on page.")
    
    for filename in FILES:
        target_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(target_path):
            print(f"[+] File already exists: {filename}")
            continue
            
        # Match URL for this filename
        matched_url = None
        clean_name = filename.replace('.zip', '')
        for url in download_links:
            if clean_name.lower() in url.lower():
                matched_url = url.replace('&amp;', '&')
                break
                
        if matched_url:
            print(f"[*] Downloading {filename}...")
            try:
                dl_req = urllib.request.Request(matched_url, headers=headers)
                with urllib.request.urlopen(dl_req) as resp, open(target_path, 'wb') as out_f:
                    total_size = int(resp.getheader('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 1024 * 1024
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            print(f"\r    {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)", end="")
                    print(f"\n[+] Completed {filename}")
            except Exception as e:
                print(f"[-] Failed to download {filename}: {e}")
        else:
            print(f"[-] Could not locate presigned link for {filename}. Ensure session cookie is valid and logged in.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download IEEE DataPort Power & EM Side-Channel Dataset")
    parser.add_argument("--session-cookie", required=True, help="Your IEEE DataPort session cookie string (e.g. 'SSESS...=...')")
    args = parser.parse_args()
    download_with_cookie(args.session_cookie)
