import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

START_URL = "https://www.connecticutchildrens.org/medical-professionals/clinical-pathways"
BASE_URL = "https://www.connecticutchildrens.org"
BASE_DOMAIN = urlparse(BASE_URL).netloc

OUTPUT_FOLDER = "data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def get_soup(url):
    """Fetch a URL and return a BeautifulSoup object."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def find_sections():
    """Extract all internal /clinical-pathways/ URLs from the main page."""
    soup = get_soup(START_URL)
    sections = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(START_URL, href)

        if urlparse(full).netloc == BASE_DOMAIN and "/clinical-pathways/" in full:
            if full != START_URL:
                sections.add(full)

    return sorted(sections)


def find_pdfs(section_url):
    """Find all PDF links in a given section page."""
    soup = get_soup(section_url)
    pdfs = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip().lower()

        if href.endswith(".pdf"):
            pdfs.add(urljoin(section_url, href))

    return sorted(pdfs)


def download_pdf(pdf_url):
    """Download a PDF to the data/ folder."""
    filename = pdf_url.split("/")[-1].split("?")[0]
    path = os.path.join(OUTPUT_FOLDER, filename)

    if os.path.exists(path):
        print(f"[skip] {filename}")
        return

    print(f"[download] {filename}")

    r = requests.get(pdf_url, timeout=20)
    if r.headers.get("content-type", "").lower().startswith("application/pdf"):
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"[saved] {filename}")
    else:
        print(f"[warning] Not a PDF content type: {pdf_url}")


def main():
    print("Collecting sections...")
    sections = find_sections()
    print(f"Found {len(sections)} sections.")

    total = 0
    for section in sections:
        print(f"\nSection: {section}")
        pdfs = find_pdfs(section)
        print(f"PDFs found: {pdfs}")

        for pdf in pdfs:
            download_pdf(pdf)
            total += 1

    print("\nDone.")
    print(f"Total PDFs downloaded: {total}")


if __name__ == "__main__":
    main()

