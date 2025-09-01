import os
import time
import re
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from PIL import Image
import PIL

class ImageScraper:
    def __init__(self, geckodriver_path="/snap/bin/geckodriver", headless=False):
        """
        Initialize the image scraper
        
        Args:
            geckodriver_path (str): Path to the geckodriver executable
            headless (bool): Whether to run Firefox in headless mode
        """
        self.geckodriver_path = geckodriver_path
        self.headless = headless
        
    def search_and_save_html(self, query, target_path):
        """
        Search for images on Google and save the HTML content
        
        Args:
            query (str): Search query
            target_path (str): Path where to save the HTML file
        """
        print(f"Searching for: {query}")
        
        # Configure Firefox options to avoid detection
        options = Options()
        options.headless = self.headless
        
        # Add user agent and other options to avoid detection
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Configure service with GeckoDriver path
        service = Service(executable_path=self.geckodriver_path)
        
        # Create Firefox browser instance
        driver = webdriver.Firefox(service=service, options=options)
        
        try:
            # Remove automation indicators
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Add random delay before starting
            time.sleep(2)
            
            # Perform Google image search with encoded query
            search_query = query.replace(" ", "+")
            search_url = f"https://www.google.com/search?q={search_query}&tbm=isch&source=hp&biw=1920&bih=1080"
            
            print(f"Accessing URL: {search_url}")
            driver.get(search_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Check if we got blocked or redirected
            current_url = driver.current_url
            page_title = driver.title.lower()
            
            if "blocked" in page_title or "captcha" in page_title or "unusual traffic" in page_title:
                print("WARNING: Google may have blocked the request. Consider using VPN or waiting.")
            
            # Scroll down to load more images with random delays
            for i in range(5):
                driver.execute_script("window.scrollBy(0,document.body.scrollHeight)")
                time.sleep(2 + (i * 0.5))  # Increasing delay
                
                # Try to click "Show more results" button if it appears
                try:
                    show_more_button = driver.find_element("css selector", "[data-ri='1']")
                    if show_more_button:
                        driver.execute_script("arguments[0].click();", show_more_button)
                        time.sleep(2)
                except:
                    pass
            
            # Extract HTML from search page
            page_source = driver.page_source
            
            # Check if we actually got image search results
            if 'data-src="data:image' not in page_source and 'src="data:image' not in page_source:
                print("WARNING: May not have gotten proper image search results")
                print(f"Page title: {driver.title}")
                print(f"Current URL: {current_url}")
            
            # Save HTML in HTML subfolder within category folder
            html_dir = os.path.join(target_path, "HTML")
            if not os.path.exists(html_dir):
                os.makedirs(html_dir)
            
            html_file_path = os.path.join(html_dir, f"{query}.txt")
            with open(html_file_path, "w", encoding="utf-8") as file:
                file.write(page_source)
            
            print(f"HTML file saved at: {html_file_path}")
            print(f"HTML file size: {len(page_source)} characters")
            return html_file_path
            
        except Exception as e:
            print(f"Error during web scraping: {e}")
            return None
        finally:
            driver.quit()
    
    def extract_and_download_images(self, html_file_path, category, images_path):
        """
        Extract image URLs from HTML and download them
        
        Args:
            html_file_path (str): Path to the HTML file
            category (str): Category name for naming images
            images_path (str): Path where to save downloaded images
        """
        print(f"Extracting and downloading images for: {category}")
        
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        
        # Read HTML content
        with open(html_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Multiple regex patterns to catch different image URL formats
        patterns = [
            r'data-src="(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))"',  # data-src URLs
            r'src="(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))"',       # regular src URLs
            r'"ou":"(https?://[^"]+\.(?:jpg|jpeg|png|gif|webp))"',      # Google Images specific
            r'imgurl=(https?://[^&]+\.(?:jpg|jpeg|png|gif|webp))',      # URL parameters
            r'<img[^>]+src="([^"]*data:image[^"]*)"'                    # base64 images
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            all_matches.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in all_matches:
            if match not in seen and match.startswith(('http', 'data:')):
                seen.add(match)
                unique_matches.append(match)
        
        print(f"Found {len(unique_matches)} unique image URLs")
        
        downloaded_count = 0
        if unique_matches:
            for idx, url in enumerate(unique_matches):
                try:
                    if url.startswith('data:'):
                        # Handle base64 encoded images
                        import base64
                        header, data = url.split(',', 1)
                        image_data = base64.b64decode(data)
                        
                        image_filename = os.path.join(images_path, f"{category}_{idx + 1}.png")
                        with open(image_filename, 'wb') as image_file:
                            image_file.write(image_data)
                    else:
                        # Handle regular HTTP URLs
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Referer': 'https://www.google.com/'
                        }
                        
                        response = requests.get(url, timeout=15, headers=headers, stream=True)
                        response.raise_for_status()
                        
                        # Check if it's actually an image
                        content_type = response.headers.get('content-type', '').lower()
                        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                            print(f"Skipping non-image content: {content_type}")
                            continue
                        
                        # Determine file extension
                        if 'jpeg' in content_type or 'jpg' in content_type:
                            ext = 'jpg'
                        elif 'png' in content_type:
                            ext = 'png'
                        elif 'gif' in content_type:
                            ext = 'gif'
                        elif 'webp' in content_type:
                            ext = 'webp'
                        else:
                            ext = 'jpg'  # default
                        
                        image_filename = os.path.join(images_path, f"{category}_{idx + 1}.{ext}")
                        with open(image_filename, 'wb') as image_file:
                            for chunk in response.iter_content(chunk_size=8192):
                                image_file.write(chunk)
                    
                    print(f"Image downloaded: {image_filename}")
                    downloaded_count += 1
                    
                    # Add small delay between downloads
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error downloading image from {url[:100]}...: {e}")
                    continue
        else:
            print(f"No images found in {html_file_path}")
        
        print(f"Total images downloaded for {category}: {downloaded_count}")
        return downloaded_count
    
    def clean_corrupted_images(self, directory):
        """
        Verify and delete corrupted images in the directory
        
        Args:
            directory (str): Directory to clean
        """
        print(f"Cleaning corrupted images in: {directory}")
        deleted_files_count = 0
        
        # Walk through all files in directory and subdirectories
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(subdir, file)
                try:
                    # Try to open and verify the image
                    with Image.open(filepath) as img:
                        img.verify()  # Verify image file integrity
                except (IOError, SyntaxError, PIL.UnidentifiedImageError) as e:
                    print(f'Deleting corrupted file: {filepath}')
                    os.remove(filepath)
                    deleted_files_count += 1
        
        print(f'Cleaning completed. Total files deleted: {deleted_files_count}')
        return deleted_files_count
    
    def scrape_images(self, categories, num_images_per_category=100, base_path="imagenes_dataset", data_path="data", use_alternative_sources=True):
        """
        Complete image scraping workflow for all categories
        
        Args:
            categories (list): List of search queries/categories
            num_images_per_category (int): Target number of images per category
            base_path (str): Base path for storing HTML files
            data_path (str): Base path for storing downloaded images
            use_alternative_sources (bool): Whether to try alternative sources if Google fails
        """
        print(f"Starting image scraping for {len(categories)} categories...")
        print("NOTE: If Google blocks requests, consider:")
        print("1. Using a VPN")
        print("2. Adding longer delays between requests")
        print("3. Using alternative image sources")
        
        # Create base directories if they don't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        total_downloaded = 0
        total_cleaned = 0
        
        for category in categories:
            print(f"\n{'='*50}")
            print(f"Processing category: {category}")
            print(f"{'='*50}")
            
            # Step 1: Search and save HTML
            category_path = os.path.join(base_path, category)
            if not os.path.exists(category_path):
                os.makedirs(category_path)
            
            html_file_path = self.search_and_save_html(category, category_path)
            
            if html_file_path is None:
                print(f"Skipping {category} due to search failure")
                continue
            
            # Step 2: Extract and download images
            images_path = os.path.join(data_path, category)
            downloaded = self.extract_and_download_images(html_file_path, category, images_path)
            
            # If no images downloaded and alternative sources enabled
            if downloaded == 0 and use_alternative_sources:
                print(f"No images downloaded for {category}. This might be due to:")
                print("- Google blocking the request")
                print("- Changed HTML structure")
                print("- Network issues")
                print("\nSuggestions:")
                print("1. Check the saved HTML file to see what was actually retrieved")
                print("2. Try running with headless=False to see what happens in the browser")
                print("3. Consider using alternative image sources or APIs")
            
            total_downloaded += downloaded
            
            # Step 3: Clean corrupted images (only if we have images)
            if downloaded > 0:
                cleaned = self.clean_corrupted_images(images_path)
                total_cleaned += cleaned
            else:
                cleaned = 0
            
            print(f"Category '{category}' completed: {downloaded} downloaded, {cleaned} cleaned")
        
        print(f"\n{'='*50}")
        print(f"SCRAPING COMPLETED")
        print(f"{'='*50}")
        print(f"Total images downloaded: {total_downloaded}")
        print(f"Total corrupted images cleaned: {total_cleaned}")
        print(f"Final images count: {total_downloaded - total_cleaned}")
        
        if total_downloaded == 0:
            print("\nTROUBLESHOoting:")
            print("- Try setting headless=False to see browser activity")
            print("- Check if you need to solve CAPTCHAs manually")
            print("- Consider using a different IP/VPN")
            print("- Use alternative image sources (Bing, DuckDuckGo, etc.)")

def main():
    # Define categories to scrape
    categories = [
        'buy yellow rubber duck', 
        'yellow rubber duck toy', 
        'yellow rubber duck white background front'
    ]
    
    # Initialize scraper
    scraper = ImageScraper(
        geckodriver_path="/snap/bin/geckodriver",
        headless=False  # Set to True for headless mode
    )
    
    # Run complete scraping workflow
    scraper.scrape_images(
        categories=categories,
        num_images_per_category=100,
        base_path="imagenes_dataset",
        data_path="data"
    )

if __name__ == "__main__":
    main()