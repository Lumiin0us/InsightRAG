import requests
from bs4 import BeautifulSoup
import random
import re
import json

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

url = "https://docs.anthropic.com/en"
PAGES_BASE_URL = 'https://platform.claude.com'

NOISE_STRINGS = {
    "was this page helpful?",
    "copy page",
    "copy link to clipboard",
    "learn more",
    "go to the quickstart",
    "read the messages api guide",
    "see the models overview",
    "browse the features overview",
}

def getAnchors(url):
    response = requests.get(url, headers={"User-Agent": random.choice(USER_AGENTS)})
    soup = BeautifulSoup(response.text, "html.parser")
    div = soup.find("div", class_=lambda c: c and "overflow-y-auto" in c and "pb-4" in c)
    anchors = div.find_all("a", "")
    links = []
    for a in anchors:
        href = a.get("href", "")
        links.append(href)
    return links

def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower() in NOISE_STRINGS:
            continue
        if re.match(r'^[.\-•:,!?]+$', line):
            continue
        if re.match(r'^\d+$', line):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()

def getPages(url, anchors):
    results = []
    for a in anchors:
        try:
            response = requests.get(url + a, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article", id="content-container")
            heading = soup.find("h1")
            if article:
                for tag in article.find_all(["script", "style", "svg", "button", "nav"]):
                    tag.decompose()
                raw_text = article.get_text(separator="\n", strip=True)
                cleaned = clean_text(raw_text)
                results.append({"heading": heading.get_text(strip=True) if heading else "", "url": url + a, "text": cleaned})
                print(f"Scraped: {url + a}")
        except Exception as e:
            print(f"Error on {url + a}: {e}")
    return results

anchors = getAnchors(url=url)
pages = getPages(PAGES_BASE_URL, anchors=anchors)

with open("anthropic_docs.json", "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2, ensure_ascii=False)

print(f"Saved {len(pages)} pages to anthropic_docs.json")