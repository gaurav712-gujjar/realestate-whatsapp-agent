"""
Mark a property as sold (or put it back as available) so it stops (or
resumes) showing up to buyers on WhatsApp. Sold-out properties are never
shown to users - every search already filters on status = 'available'.

Usage:
    python mark_property_status.py --id 42 --status sold
    python mark_property_status.py --name "Sunrise Apts" --status sold
    python mark_property_status.py --id 42 --status available
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.agent import RealEstateAgent  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, help="Property id (from the properties table)")
    parser.add_argument("--name", type=str, help="Property name (partial match) if id is unknown")
    parser.add_argument("--status", choices=["sold", "available"], required=True)
    args = parser.parse_args()

    if not args.id and not args.name:
        parser.error("Provide --id or --name.")

    agent = RealEstateAgent()
    updated = agent.mark_property_status(property_id=args.id, property_name=args.name, status=args.status)
    if updated:
        print(f"✅ Updated {updated} row(s) to status='{args.status}'.")
    else:
        print("⚠️  No matching property found - nothing was updated.")


if __name__ == "__main__":
    main()