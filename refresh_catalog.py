#!/usr/bin/env python3
"""
Standalone script to scrape the SHL catalog and save it to data/catalog.json.
Run this whenever you want to refresh the catalog:

    python refresh_catalog.py [--descriptions]

--descriptions  Also fetch description from each product page (slower, ~5 min)
"""

import sys
import argparse
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from catalog_scraper import scrape_full_catalog, save_catalog, load_catalog, DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Refresh the SHL assessment catalog.")
    parser.add_argument(
        "--descriptions",
        action="store_true",
        help="Also scrape product description pages (slower)",
    )
    args = parser.parse_args()

    print("Starting SHL catalog scrape...")
    print(f"Target: https://www.shl.com/solutions/products/product-catalog/")
    print(f"Output: {DATA_PATH}\n")

    products = scrape_full_catalog(enrich_descriptions=args.descriptions)

    if not products:
        print("⚠️  Scraper returned 0 products.")
        print("   This can happen if SHL's site structure has changed.")
        print("   The bundled data/catalog.json fallback will be used instead.")
        
        existing = load_catalog()
        if existing:
            print(f"   Existing catalog has {len(existing)} assessments — keeping it.")
        sys.exit(1)

    save_catalog(products)
    print(f"\n✅ Saved {len(products)} assessments to {DATA_PATH}")

    # Print summary by test type
    from collections import Counter
    types = Counter()
    for p in products:
        for t in p.get("test_type", "A").split(","):
            types[t.strip()] += 1
    print("\nBreakdown by test type:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
