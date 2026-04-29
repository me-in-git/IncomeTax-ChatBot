import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Settings
DOWNLOAD_FOLDER = "downloaded_pdfs"
INCOME_TAX_URLS = [
    "https://www.incometax.gov.in",
    "https://www.incometaxindia.gov.in",
]

# Create download folder
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Track downloaded PDFs to avoid duplicates
downloaded_pdfs = set()

def get_all_links(url, max_pages=100):
    """Recursively get all links from a website"""
    links = set()
    to_visit = {url}
    visited = set()
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
            
        try:
            print(f"Crawling: {current_url}")
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            visited.add(current_url)
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)
                
                # Only stay within the same domain
                if urlparse(full_url).netloc == urlparse(url).netloc:
                    # Add PDF links
                    if full_url.lower().endswith('.pdf'):
                        links.add(full_url)
                    # Add other pages to visit
                    elif full_url not in visited and len(visited) < max_pages:
                        to_visit.add(full_url)
                        
            time.sleep(0.5)  # Be respectful to the server
            
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            
    return links

def download_pdf(pdf_url, folder):
    """Download a single PDF"""
    try:
        # Create a safe filename
        filename = pdf_url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename = hashlib.md5(pdf_url.encode()).hexdigest() + '.pdf'
        
        filepath = os.path.join(folder, filename)
        
        # Check if already downloaded
        if filename in downloaded_pdfs:
            return None
            
        print(f"Downloading: {filename}")
        response = requests.get(pdf_url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = (downloaded / total_size) * 100
                    print(f"  Progress: {progress:.1f}%", end='\r')
        
        downloaded_pdfs.add(filename)
        print(f"\n✅ Downloaded: {filename} ({total_size/1024/1024:.2f} MB)")
        return filepath
        
    except Exception as e:
        print(f"❌ Failed to download {pdf_url}: {e}")
        return None

def crawl_and_download_all_pdfs(base_urls, max_pages_per_site=50):
    """Main function to crawl and download all PDFs"""
    
    all_pdf_links = set()
    
    # Step 1: Crawl websites to find all PDF links
    print("="*60)
    print("STEP 1: Crawling websites for PDF links...")
    print("="*60)
    
    for url in base_urls:
        print(f"\n🌐 Crawling: {url}")
        pdf_links = get_all_links(url, max_pages=max_pages_per_site)
        all_pdf_links.update(pdf_links)
        print(f"   Found {len(pdf_links)} PDF links")
    
    print(f"\n📊 Total unique PDFs found: {len(all_pdf_links)}")
    
    # Step 2: Download all PDFs
    print("\n" + "="*60)
    print("STEP 2: Downloading PDFs...")
    print("="*60)
    
    downloaded_count = 0
    failed_count = 0
    
    # Download sequentially (to avoid overwhelming server)
    for pdf_url in all_pdf_links:
        result = download_pdf(pdf_url, DOWNLOAD_FOLDER)
        if result:
            downloaded_count += 1
        else:
            failed_count += 1
        time.sleep(0.5)  # Delay between downloads
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print(f"✅ Downloaded: {downloaded_count} PDFs")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Saved to: {DOWNLOAD_FOLDER}")
    
    return downloaded_count

# Alternative: Use sitemap to find PDFs (more efficient)
def get_pdfs_from_sitemap(sitemap_url):
    """Extract PDF links from sitemap.xml"""
    pdf_links = set()
    
    try:
        response = requests.get(sitemap_url, timeout=10)
        soup = BeautifulSoup(response.text, 'xml')
        
        # Find all loc tags
        for loc in soup.find_all('loc'):
            url = loc.text
            if url.lower().endswith('.pdf'):
                pdf_links.add(url)
                
    except Exception as e:
        print(f"Could not parse sitemap: {e}")
    
    return pdf_links

# Specialized search for tax-related PDFs
def search_tax_pdfs():
    """Search for specific tax-related PDFs using patterns"""
    
    # Common PDF patterns on Income Tax website
    pdf_patterns = [
        "https://www.incometaxindia.gov.in/Documents/",
        "https://www.incometaxindia.gov.in/Forms/",
        "https://www.incometaxindia.gov.in/Tax%20Tools/",
        "https://incometaxindia.gov.in/Downloads/",
        "https://www.incometax.gov.in/iec/foportal/",
    ]
    
    all_pdfs = set()
    
    for base_url in pdf_patterns:
        try:
            print(f"Checking: {base_url}")
            response = requests.get(base_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf'):
                    full_url = urljoin(base_url, href)
                    all_pdfs.add(full_url)
        except:
            pass
    
    return all_pdfs

# Simple function to download from known PDF directories
def download_known_pdfs():
    """Download PDFs from known Income Tax PDF directories"""
    
    # Known PDF URLs (add more as you find them)
    known_pdf_urls = [
        "https://www.incometaxindia.gov.in/Documents/ITR%201%20Instructions%20English.pdf",
        "https://www.incometaxindia.gov.in/Documents/ITR%202%20Instructions%20English.pdf",
        "https://www.incometaxindia.gov.in/Documents/ITR%203%20Instructions%20English.pdf",
        "https://www.incometaxindia.gov.in/Documents/ITR%204%20Instructions%20English.pdf",
        "https://www.incometaxindia.gov.in/Documents/ITR%205%20Instructions%20English.pdf",
        "https://www.incometaxindia.gov.in/Documents/ITR%206%20Instructions%20English.pdf",
        "https://www.incometaxindia.gov.in/Documents/ITR%207%20Instructions%20English.pdf",
    ]
    
    for pdf_url in known_pdf_urls:
        download_pdf(pdf_url, DOWNLOAD_FOLDER)
        time.sleep(1)

# Main execution
if __name__ == "__main__":
    print("💰 INCOME TAX PDF DOWNLOADER")
    print("="*60)
    
    # Choose method:
    print("\nChoose download method:")
    print("1. Crawl entire website (slow, thorough)")
    print("2. Download from known directories (fast, limited)")
    print("3. Download specific tax forms")
    
    choice = input("\nEnter choice (1/2/3): ")
    
    if choice == "1":
        # Method 1: Full crawl
        count = crawl_and_download_all_pdfs(
            INCOME_TAX_URLS, 
            max_pages_per_site=100
        )
        
    elif choice == "2":
        # Method 2: Known directories
        print("\n📥 Downloading from known directories...")
        all_pdfs = search_tax_pdfs()
        for pdf_url in all_pdfs:
            download_pdf(pdf_url, DOWNLOAD_FOLDER)
            time.sleep(0.5)
            
    else:
        # Method 3: Specific forms
        print("\n📥 Downloading common tax forms...")
        download_known_pdfs()
    
    print(f"\n✅ All PDFs saved to: {DOWNLOAD_FOLDER}")
    print(f"📁 Total files: {len(os.listdir(DOWNLOAD_FOLDER))}")