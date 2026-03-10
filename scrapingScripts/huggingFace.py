from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import random
import json
from urllib.parse import urljoin

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

def fetch_rendered(url: str, delay: float = 3.0) -> BeautifulSoup:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 800},
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle")
        time.sleep(delay)
        html = page.content()
        browser.close()
    return BeautifulSoup(html, "html.parser")


def get_category_urls(soup: BeautifulSoup) -> list[str]:
    grids = soup.find_all("div", class_=lambda c: c and "gap-3.5" in c and "xl:grid-cols-3" in c)
    urls = []
    for grid in grids:
        for a in grid.find_all("a", href=True):
            href = a.get("href", "")
            full_url = urljoin("https://huggingface.co", href).split("#")[0]
            if full_url.startswith("https://huggingface.co/docs/") and full_url not in urls:
                urls.append(full_url)
    return urls


def get_sidebar_links(soup: BeautifulSoup, category_url: str) -> list[str]:
    """Get all sidebar links from a category page."""
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if not href or href.startswith("#") or href.startswith("http"):
            continue
        full_url = urljoin("https://huggingface.co", href).split("#")[0] # only keep links within the same doc category
        base_category = category_url.rstrip("/")
        if full_url.startswith(base_category) and full_url not in links:
            links.append(full_url)
    return links


def extract_text(soup: BeautifulSoup, url: str) -> dict:
    main = soup.find("main") or soup.find("article") or soup.body
    for tag in main.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url

    sections = []
    current_section = {"heading": title, "content": ""}
    for elem in main.find_all(["h1", "h2", "h3", "p", "pre", "li", "code"]):
        if elem.name in ("h1", "h2", "h3"):
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {"heading": elem.get_text(strip=True), "content": ""}
        else:
            current_section["content"] += elem.get_text(" ", strip=True) + "\n"

    if current_section["content"].strip():
        sections.append(current_section)

    return {"url": url, "title": title, "sections": sections}


def scrape_all():
    all_results = []
    visited = set()

    print("🔍 Fetching main docs page...")
    main_soup = fetch_rendered("https://huggingface.co/docs", delay=5.0)
    category_urls = get_category_urls(main_soup)
    print(f"Found {len(category_urls)} categories")

    for category_url in category_urls:
        print(f"\nCategory: {category_url}")

        # avoid rate limiting
        time.sleep(random.uniform(3.0, 6.0))

        category_soup = fetch_rendered(category_url, delay=4.0)
        sidebar_links = get_sidebar_links(category_soup, category_url)
        print(f"Found {len(sidebar_links)} subpages")

        all_links = [category_url] + sidebar_links
        for url in all_links:
            if url in visited:
                continue
            visited.add(url)

            try:
                time.sleep(random.uniform(2.0, 4.0))

                page_soup = fetch_rendered(url, delay=3.0)
                page_data = extract_text(page_soup, url)
                all_results.append(page_data)
                print(f"{page_data['title']} ({len(page_data['sections'])} sections)")

            except Exception as e:
                print(f"Error on {url}: {e}")

    return all_results


if __name__ == "__main__":
    data = scrape_all()
    with open("huggingface_docs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(data)} pages to huggingface_docs.json")