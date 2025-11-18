import os
import requests
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Disable warnings for optional SSL fallback
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

START_URL = "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways"
BASE_URL = "https://www.connecticutchildrens.org"
BASE_DOMAIN = urlparse(BASE_URL).netloc

OUTPUT_FOLDER = "clinical_pathways_pdfs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def download_pdf(url):
    filename = url.split("/")[-1]
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    # Do not re-download files
    if os.path.exists(filepath):
        print(f"Skipping (already exists): {filename}")
        return

    print(f"Downloading: {filename}")

    # First try normal SSL
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception:
        print(f"SSL error, retrying without verification: {url}")
        try:
            r = requests.get(url, verify=False, timeout=15)
            r.raise_for_status()
        except Exception as e2:
            print(f"Failed to download {url}: {e2}")
            return

    with open(filepath, "wb") as f:
        f.write(r.content)


def get_soup(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to load page: {url} ({e})")
        return None
    return BeautifulSoup(r.text, "html.parser")


def get_section_links():
    soup = get_soup(START_URL)
    if soup is None:
        return []

    section_links = []

    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        parsed = urlparse(href)

        # Only internal pages
        if parsed.netloc == BASE_DOMAIN:
            # Only actual clinical pathways pages
            if "/clinical-pathways/" in parsed.path:
                section_links.append(href)

    return sorted(list(set(section_links)))


def get_pdfs_from_section(url):
    soup = get_soup(url)
    if soup is None:
        return []

    pdf_links = []

    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        parsed = urlparse(href)

        # Only internal PDFs
        if parsed.netloc == BASE_DOMAIN and href.lower().endswith(".pdf"):
            pdf_links.append(href)

    return list(set(pdf_links))


def main():
    print("Collecting clinical pathway sections...")
    sections = get_section_links()
    print(f"Found {len(sections)} sections.")

    total_pdfs = 0

    for section in sections:
        print(f"\nChecking section: {section}")
        pdfs = get_pdfs_from_section(section)

        if not pdfs:
            print("No PDFs found in this section.")
            continue

        print(f"Found {len(pdfs)} PDF(s).")

        for pdf in pdfs:
            download_pdf(pdf)
            total_pdfs += 1

    print("\nDone.")
    print(f"Total PDFs processed (downloaded or already present): {total_pdfs}")


if __name__ == "__main__":
    main()
