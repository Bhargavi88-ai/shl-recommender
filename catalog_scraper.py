"""
SHL Product Catalog Scraper
Scrapes Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
"""

import httpx
from bs4 import BeautifulSoup
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
DATA_PATH = Path(__file__).parent / "data" / "catalog.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Test type mapping based on SHL's letter codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Situational Judgement",
}


def scrape_catalog_page(start: int = 0, type_filter: int = 1) -> List[Dict[str, Any]]:
    """Scrape a single page of the SHL catalog."""
    params = f"?start={start}&type={type_filter}"
    url = CATALOG_URL + params

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers=HEADERS)
            resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch catalog page (start={start}): {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    products = []

    # SHL catalog uses a table structure
    # Try multiple selectors to be robust across page layout changes
    rows = (
        soup.select("table.product-table tbody tr")
        or soup.select(".product-catalogue table tbody tr")
        or soup.select("table tbody tr")
    )

    for row in rows:
        try:
            cells = row.find_all("td")
            if not cells:
                continue

            # First cell: product name + link
            name_cell = cells[0]
            link_el = name_cell.find("a")
            if not link_el:
                continue

            name = link_el.get_text(strip=True)
            href = link_el.get("href", "")
            if href and not href.startswith("http"):
                href = BASE_URL + href

            if not name or not href:
                continue

            # Remote testing (column 1 or 2)
            remote_testing = False
            adaptive_irt = False
            test_types = []

            # Parse remaining cells for flags and test types
            for cell in cells[1:]:
                cell_text = cell.get_text(strip=True)
                # Check for filled/empty circle indicators
                imgs = cell.find_all("img")
                has_check = any(
                    "yes" in (img.get("alt", "") + img.get("src", "")).lower()
                    or "check" in (img.get("alt", "") + img.get("src", "")).lower()
                    for img in imgs
                )
                spans = cell.find_all("span")
                for span in spans:
                    cls = " ".join(span.get("class", []))
                    txt = span.get_text(strip=True)
                    if txt in TEST_TYPE_MAP:
                        test_types.append(txt)
                    if "remote" in cls.lower() and ("yes" in txt.lower() or has_check):
                        remote_testing = True
                    if "adaptive" in cls.lower() and ("yes" in txt.lower() or has_check):
                        adaptive_irt = True

            # Fallback: check data attributes
            if not test_types:
                for cell in cells[1:]:
                    for attr in cell.attrs:
                        if "type" in attr.lower():
                            val = cell[attr]
                            if val in TEST_TYPE_MAP:
                                test_types.append(val)

            products.append(
                {
                    "name": name,
                    "url": href,
                    "test_type": ", ".join(test_types) if test_types else "A",
                    "test_type_labels": [TEST_TYPE_MAP.get(t, t) for t in test_types],
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt,
                    "description": "",
                }
            )

        except Exception as e:
            logger.warning(f"Error parsing row: {e}")
            continue

    return products


def scrape_product_detail(url: str) -> str:
    """Fetch a brief description from the product detail page."""
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(url, headers=HEADERS)
            resp.raise_for_status()
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try common description selectors
    for sel in [
        ".product-description",
        ".hero__description",
        ".intro__text",
        "meta[name='description']",
        ".content-block p",
    ]:
        el = soup.select_one(sel)
        if el:
            if el.name == "meta":
                return el.get("content", "")[:500]
            text = el.get_text(strip=True)
            if len(text) > 50:
                return text[:500]

    return ""


def scrape_full_catalog(enrich_descriptions: bool = False) -> List[Dict[str, Any]]:
    """
    Scrape all Individual Test Solutions from the SHL catalog.
    Paginates through all pages.
    """
    all_products = []
    start = 0
    page_size = 12
    max_pages = 100  # Safety limit

    logger.info("Starting SHL catalog scrape (Individual Test Solutions only)...")

    for _ in range(max_pages):
        logger.info(f"Fetching page starting at {start}...")
        page_products = scrape_catalog_page(start=start, type_filter=1)

        if not page_products:
            logger.info(f"No more products found at start={start}, stopping.")
            break

        all_products.extend(page_products)
        logger.info(f"Found {len(page_products)} products (total: {len(all_products)})")

        start += page_size
        time.sleep(0.8)  # Be polite to SHL's server

    # Optionally enrich with descriptions
    if enrich_descriptions and all_products:
        logger.info("Enriching products with descriptions...")
        for i, product in enumerate(all_products):
            if product.get("url"):
                desc = scrape_product_detail(product["url"])
                if desc:
                    all_products[i]["description"] = desc
            time.sleep(0.5)

    logger.info(f"Scraping complete. Total products: {len(all_products)}")
    return all_products


def save_catalog(products: List[Dict[str, Any]], path: Path = DATA_PATH) -> None:
    """Save catalog to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(products)} products to {path}")


def load_catalog(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    """Load catalog from JSON file."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    products = scrape_full_catalog(enrich_descriptions=True)
    if products:
        save_catalog(products)
        print(f"Saved {len(products)} assessments to {DATA_PATH}")
    else:
        print("No products scraped – check network/selectors.")
