import sqlite3
import os
import json
import csv
import re
import time
import logging
import threading
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("real_estate_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# A module-level lock so concurrent requests within the same process never
# interleave writes to the same CSV file (Python's GIL doesn't protect
# multi-step file I/O on its own).
_CSV_WRITE_LOCK = threading.Lock()

MAX_MESSAGE_LENGTH = 1000          # guard against abuse / runaway LLM cost
MAX_CONSECUTIVE_FAILURES = 2        # after this many failed extractions in a row, escalate to a human
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "").strip()


class IntentExtraction(BaseModel):
    intent: str = Field(description="One of: 'buy', 'sell', 'continue_buy', 'continue_sell', 'select_property', 'greeting', or 'inquiry'. Use 'select_property' ONLY when the user is confirming/choosing one of the properties that was just shown to them (e.g. 'I choose this one', 'ye wala theek hai', 'book this one', 'I'm interested in this'), as opposed to 'continue_buy' which means they want to see more/different options or are still giving search criteria.")
    params: dict = Field(description="The extracted parameters for the intent")
    language: str = Field(description="The language style of the user's message: either 'english' (plain English) or 'hinglish' (Hindi/Hinglish, written in Roman or Devanagari script)")

# Static prompts in both supported languages, keyed by field name / message name.
TEXTS = {
    "greeting": {
        "hinglish": "Namaste! Main aapka Real Estate AI agent hoon. 🏠 Main aapki property buy ya sell karne mein poori help kar sakta hoon. Bataiye, aap property buy karna chahte hain ya sell?",
        "english": "Hello! I'm your Real Estate AI agent. 🏠 I can help you buy or sell a property. Would you like to buy or sell?",
    },
    "inquiry": {
        "hinglish": "Ji, hamare paas Flats, Plots aur Houses/Villas ki bahut saari exclusive listings hain Jaipur aur aas-paas ki prime locations mein. Aap kis type ki property dekh rahe hain?",
        "english": "We have a wide range of exclusive Flats, Plots, and Houses/Villas listed across Jaipur and nearby prime locations. What type of property are you looking for?",
    },
    "fallback": {
        "hinglish": "Ji, main aapki kaise madad kar sakta hoon? Property buy karni hai ya sell?",
        "english": "How can I help you today? Would you like to buy or sell a property?",
    },
    "extraction_failed": {
        "hinglish": "Ji, main thoda samajh nahi paaya. Kya aap dobara bol sakte hain?",
        "english": "Sorry, I didn't quite catch that. Could you please repeat it?",
    },
    "escalate": {
        "hinglish": "Lagta hai main is baar aapki baat sahi se samajh nahi pa raha hoon. Main aapko hamari team se connect kar deta hoon jo turant help karegi.",
        "english": "It looks like I'm having trouble understanding that. Let me connect you with our team directly so they can help you right away.",
    },
    "sell_done": {
        "hinglish": "Bahut badhiya! Maine saari details note kar li hain. Hamari team jaldi hi aapse contact karegi. Thank you!",
        "english": "Great! I've noted down all the details. Our team will get in touch with you shortly. Thank you!",
    },
    "exact_intro": {
        "hinglish": "Ji, aapki requirement ke hisaab se ye options hain:",
        "english": "Here are some options that match what you're looking for:",
    },
    "fallback_intro": {
        "hinglish": "Ji, aapke exact budget/location mein toh available nahi hai, lekin ye closest options hain jo aapko pasand aa sakte hain:",
        "english": "We don't have an exact match for that budget/location, but here are the closest options you might like:",
    },
    "fallback_intro_location": {
        "hinglish": "Ji, is location mein abhi hamare paas koi property available nahi hai. Lekin ghabraiye mat — same type aur budget mein humare paas ye zabardast options doosri prime locations mein hain, ye bhi dekh lijiye:",
        "english": "We don't currently have anything available in that specific location. But don't worry — here are some great options of the same type and budget in other prime locations you should definitely consider:",
    },
    "fallback_intro_budget": {
        "hinglish": "Ji, aapke exact budget mein is location mein available nahi hai, lekin thoda upar ki range mein ye options hain jo dekhne layak hain:",
        "english": "We don't have anything in that exact budget for this location, but here are some slightly above-budget options in the same area that are well worth considering:",
    },
    "more_options_cta": {
        "hinglish": "Agar inme se koi pasand aaye toh site visit book karwa dete hain! Aur options ke liye 'aur dikhao' bol dijiye.",
        "english": "Let me know if you'd like to book a site visit for any of these! You can also just say 'show more options' for more choices.",
    },
    "exhausted": {
        "hinglish": "Ji, jo bhi listings aapki requirement se match karti thi, wo maine already dikha di hain. Agar aap budget ya location thoda flexible kar dein, toh main aur options dhoond sakta hoon.",
        "english": "I've already shown you every listing that matched your requirement. If you can be a little flexible on budget or location, I can look for more options.",
    },
    "no_listings_at_all": {
        "hinglish": "Ji, abhi is criteria se matching koi property available nahi hai. Main apna offline network check karke 1-2 din mein aapko update karta hoon. Kya aap budget ya location thoda flexible kar sakte hain, taaki main kuch options dhoond sakoon?",
        "english": "We don't have anything matching that criteria right now. Let me check our offline network and get back to you within 1-2 days. Could you share any flexibility on budget or location so I can look for something in the meantime?",
    },
    "selection_confirmed": {
        "hinglish": "Bahut badhiya choice hai! 🎉 Ye property genuinely ek achha option hai — location, price aur features sab kaafi solid hain. Maine aapki details note kar li hain, hamari team bahut jaldi aapse contact karke site visit aur aage ki process finalize kar degi. Aap sahi haathon mein hain, tension mat lijiye!",
        "english": "Excellent choice! 🎉 This is genuinely a great pick — solid location, price, and features. I've noted down your details, and our team will reach out to you very shortly to arrange a site visit and take things forward. You're in good hands!",
    },
    "clarify_selection": {
        "hinglish": "Aapne kaunsi property choose ki hai? Kripya naam ya number (1, 2, 3) batayein taaki main sahi property ke liye aapki details note kar sakoon:",
        "english": "Which one of these are you choosing? Please tell me the name or number (1, 2, 3) so I note down your interest against the correct property:",
    },
    "empty_message": {
        "hinglish": "Ji, aapka message khaali aaya. Kya aap dobara bata sakte hain aapko kya chahiye?",
        "english": "It looks like your message came through empty. Could you tell me again what you're looking for?",
    },
    "unexpected_error": {
        "hinglish": "Maaf kijiye, kuch technical dikkat aa gayi. Kripya thodi der baad dobara try karein, ya hamari team se seedha baat karein.",
        "english": "Sorry, something went wrong on our end. Please try again in a moment, or reach out to our team directly.",
    },
}

REQUIRED_BUY = {
    "property_type": {
        "hinglish": "Aapko kis tarah ki property chahiye? (Flat, Plot, ya Villa?)",
        "english": "What type of property are you looking for? (Flat, Plot, or Villa?)",
    },
    "location": {
        "hinglish": "Aap kaunsi location mein property dekh rahe hain?",
        "english": "Which location are you looking for a property in?",
    },
    "max_price": {
        "hinglish": "Aapka budget kitna hai? (e.g., 50 Lakhs)",
        "english": "What is your budget? (e.g., 50 Lakhs)",
    },
}

REQUIRED_SELL = {
    "property_type": {
        "hinglish": "Ji zaroor! Pehle bataiye property kis type ki hai? (Flat, Plot, ya House?)",
        "english": "Sure! First, tell me what type of property it is (Flat, Plot, or House?)",
    },
    "property_name": {
        "hinglish": "Building ya project ka naam kya hai?",
        "english": "What is the name of the building or project?",
    },
    "location": {
        "hinglish": "Property ki exact location kya hai?",
        "english": "What is the exact location of the property?",
    },
    "price": {
        "hinglish": "Aap kitna price expect kar rahe hain?",
        "english": "What price are you expecting?",
    },
    "area": {
        "hinglish": "Total area kitna hai sq-ft mein?",
        "english": "What is the total area in sq-ft?",
    },
    "face": {
        "hinglish": "Property kis side facing hai? (e.g., East, West)",
        "english": "Which direction does the property face? (e.g., East, West)",
    },
    "outside_view": {
        "hinglish": "Bahar ka view kaisa hai?",
        "english": "What's the view like from outside?",
    },
    "age": {
        "hinglish": "Property kitni puraani hai?",
        "english": "How old is the property?",
    },
    "description": {
        "hinglish": "Thoda property ke baare mein bataiye (features, etc.)",
        "english": "Tell me a bit about the property (features, etc.)",
    },
    "images": {
        "hinglish": "Agar images hain toh unka link share kar dijiye.",
        "english": "If you have any images, please share the link.",
    },
}

# Asked once per session, before the buy/sell details, so every lead/listing
# we save actually has a real name and phone number attached to it.
REQUIRED_CONTACT = {
    "user_name": {
        "hinglish": "Sabse pehle, aapka naam jaan sakta hoon?",
        "english": "First, may I know your name?",
    },
    "user_phone": {
        "hinglish": "Dhanyavaad! Aur aapka contact number kya hai, taaki hamari team aapse directly baat kar sake?",
        "english": "Thanks! And what's the best phone number to reach you on, so our team can follow up directly?",
    },
    "user_phone_invalid": {
        "hinglish": "Ye number sahi nahi lag raha. Kripya ek valid phone number (10 digit ya country code ke saath) share karein.",
        "english": "That doesn't look like a valid phone number. Could you share a valid number (10 digits, or with country code)?",
    },
}


class RealEstateAgent:
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not groq_key and not openai_key:
            # Fail fast and loud at startup rather than crashing (or worse,
            # silently 401'ing) on the very first message in production.
            raise RuntimeError(
                "No LLM credentials configured. Set GROQ_API_KEY or OPENAI_API_KEY "
                "in the environment before starting the agent."
            )

        if groq_key:
            self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, groq_api_key=groq_key)
        else:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini", temperature=0,
                api_key=openai_key,
                base_url=os.getenv("OPENAI_API_BASE")
            )

        # Use absolute paths or relative to the root project directory
        # Data files are now in the root/database folder
        database_dir = os.getenv("DATABASE_DIR") or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database')
        os.makedirs(database_dir, exist_ok=True)
        self.db_path = os.path.join(database_dir, 'properties.db')
        self.user_listings_path = os.path.join(database_dir, 'user_listings.csv')
        self.leads_path = os.path.join(database_dir, 'interested_leads.csv')
        self.parser = JsonOutputParser(pydantic_object=IntentExtraction)

        if not os.path.exists(self.db_path):
            logger.warning(
                "properties.db not found at %s - buy queries will find nothing until it's created "
                "(run setup_db.py).", self.db_path
            )

    # ---------- helpers ----------

    @staticmethod
    def _to_number(value):
        """Best-effort parse of user-provided price/area strings like '50 Lakhs' or '50L'."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).lower().replace(',', '').strip()
        multiplier = 1
        if 'crore' in text or text.endswith('cr'):
            multiplier = 10_000_000
        elif 'lakh' in text or text.endswith('l'):
            multiplier = 100_000
        digits = ''.join(ch for ch in text if ch.isdigit() or ch == '.')
        try:
            return float(digits) * multiplier if digits else None
        except ValueError:
            return None

    @staticmethod
    def _normalize_language(value):
        """Map whatever the LLM returns ('Hindi', 'English', 'Hinglish', etc.) to one of our two buckets."""
        if not value:
            return None
        v = str(value).strip().lower()
        if 'eng' in v and 'hin' not in v:
            return 'english'
        return 'hinglish'

    @staticmethod
    def _end_conversation(current_data):
        """
        Ends the current conversational flow. Unlike a raw session_clear, this
        explicitly returns a MINIMAL dict containing only identity info
        (name/phone/language) if we already have it, so a returning user isn't
        asked their name and number again - while guaranteeing every internal
        flow-tracking key (_flow, _pending_field, _shown_ids, _last_search,
        _resume_intent, etc.) is dropped. This is what prevents a stale flow
        state from ever leaking into a brand-new conversation.
        """
        preserved = {}
        for key in ("user_name", "user_phone", "_lang"):
            if current_data.get(key):
                preserved[key] = current_data[key]
        return preserved

    @staticmethod
    def _is_valid_phone(value):
        """Loose validation - just enough to catch obvious junk (too short,
        letters only, etc.) without being overly strict about formats."""
        if not value:
            return False
        digits = ''.join(ch for ch in str(value) if ch.isdigit())
        return 7 <= len(digits) <= 15

    @staticmethod
    def _is_wildcard(value):
        """True if the user answered a question with 'any'/'koi bhi'/'whatever' etc.
        Such an answer should count as the field being 'answered' (so we stop
        asking) but should not be used to filter the search."""
        if not value:
            return False
        v = str(value).strip().lower()
        wildcard_phrases = (
            'any', 'anything', 'whatever', 'anywhere', "doesn't matter", 'doesnt matter',
            'koi bhi', 'kuch bhi', 'kisi bhi', 'jo bhi', 'koi v', 'kuch v', 'sab chalega'
        )
        return any(phrase in v for phrase in wildcard_phrases)

    @staticmethod
    def _format_inr(value):
        """Deterministic price formatting - never left to the LLM to avoid invented figures."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        if value >= 10_000_000:
            return f"₹{value / 10_000_000:.2f} Cr"
        if value >= 100_000:
            return f"₹{value / 100_000:.2f} Lakh"
        return f"₹{value:,.0f}"

    def _row_to_dict(self, row, columns):
        return dict(zip(columns, row))

    def _build_property_caption(self, prop, lang):
        """
        Builds the property's shown text directly from the database row -
        never from the LLM - so facts (price, area, location, etc.) can
        never be hallucinated or altered.
        """
        name = prop.get('property_name') or prop.get('property_type') or 'Property'
        loc = prop.get('location') or prop.get('location_area') or ''
        price_text = self._format_inr(prop.get('total_price'))
        area = prop.get('area_sqft')
        bhk = prop.get('bhk')
        bath = prop.get('bath')
        face = prop.get('face')
        status = prop.get('property_status')
        link = prop.get('property_link')
        desc = (prop.get('property_description') or '').strip()

        lines = [f"🏠 {name}"]
        if loc:
            lines.append(f"📍 {loc}")
        if price_text:
            lines.append(f"💰 {price_text}")

        specs = []
        if bhk:
            specs.append(f"{bhk} BHK")
        if area:
            try:
                specs.append(f"{float(area):.0f} sqft")
            except (TypeError, ValueError):
                specs.append(f"{area} sqft")
        if bath:
            specs.append(f"{bath} Bath")
        if specs:
            lines.append(" | ".join(specs))

        if face:
            lines.append(f"Facing: {face}")
        if status:
            lines.append(f"Status: {status}")
        if desc:
            lines.append(desc[:200])
        if link:
            label = "More info" if lang == "english" else "Zyada jaankari"
            lines.append(f"{label}: {link}")

        return "\n".join(lines)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        # WAL mode lets reads and writes coexist without locking each other out -
        # important once this is behind multiple gunicorn workers.
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error:
            pass
        return conn

    def query_properties(self, prop_type=None, location=None, max_price=None, exclude_ids=None):
        """
        Runs the actual search. Any DB error (missing file, locked file,
        corrupt schema) is caught and logged - callers always get back an
        empty result set instead of an unhandled exception blowing up the
        whole message-handling request.
        """
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = "SELECT * FROM properties WHERE status = 'available'"
            params = []
            if prop_type:
                query += " AND property_type LIKE ?"
                params.append(f"%{prop_type}%")
            if location:
                query += " AND (location LIKE ? OR location_area LIKE ?)"
                params.extend([f"%{location}%", f"%{location}%"])
            numeric_max_price = self._to_number(max_price)
            if numeric_max_price is not None:
                query += " AND total_price <= ?"
                params.append(numeric_max_price)
            if exclude_ids:
                placeholders = ",".join("?" * len(exclude_ids))
                query += f" AND id NOT IN ({placeholders})"
                params.extend(exclude_ids)
            query += " LIMIT 3"
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            logger.debug(
                "query_properties(type=%r, location=%r, max_price=%r, exclude=%d ids) -> %d row(s) "
                "[sold-out/unavailable properties are excluded via status='available' filter]",
                prop_type, location, max_price, len(exclude_ids or []), len(results)
            )
            return results, columns
        except sqlite3.Error as e:
            logger.error("Database error while querying properties: %s", e, exc_info=True)
            return [], []
        finally:
            if conn is not None:
                conn.close()

    def mark_property_status(self, property_id=None, property_name=None, status="sold"):
        """
        Flips a property's availability. Once marked 'sold', it's automatically
        excluded from every search (query_properties always filters on
        status = 'available'), so this is the mechanism for keeping sold-out
        units from ever being shown to a buyer again.
        Match by property_id (preferred, exact) or property_name (LIKE match)
        if the id isn't known. Returns the number of rows updated.
        """
        if not property_id and not property_name:
            raise ValueError("Provide either property_id or property_name.")
        if status not in ("available", "sold"):
            raise ValueError("status must be 'available' or 'sold'.")

        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            if property_id:
                cursor.execute("UPDATE properties SET status = ? WHERE id = ?", (status, property_id))
            else:
                cursor.execute("UPDATE properties SET status = ? WHERE property_name LIKE ?", (status, f"%{property_name}%"))
            conn.commit()
            updated = cursor.rowcount
            logger.info(
                "Marked %d propert(y/ies) as '%s' (id=%s, name=%s).",
                updated, status, property_id, property_name
            )
            return updated
        except sqlite3.Error as e:
            logger.error("Failed to update property status: %s", e, exc_info=True)
            return 0
        finally:
            if conn is not None:
                conn.close()

    def find_best_properties(self, prop_type=None, location=None, max_price=None, exclude_ids=None):
        """
        Try an exact match first, then progressively relax constraints (budget,
        then location, then type) so we almost always have *something* to show
        the user instead of turning them away empty-handed.
        Returns (results, columns, exact_match: bool, relaxed_fields: list[str]).
        relaxed_fields tells the caller exactly which of the user's original
        criteria had to be dropped to find these results (e.g. ['location']
        means "nothing in that location, but we found matches elsewhere"),
        so the response can be worded specifically instead of generically.
        """
        original = {"property_type": prop_type, "location": location, "max_price": max_price}
        attempts = [
            dict(prop_type=prop_type, location=location, max_price=max_price),
            dict(prop_type=prop_type, location=location, max_price=None),   # relax budget
            dict(prop_type=prop_type, location=None, max_price=max_price),  # relax location
            dict(prop_type=prop_type, location=None, max_price=None),       # keep only type
            dict(prop_type=None, location=location, max_price=None),        # keep only location
            dict(prop_type=None, location=None, max_price=None),            # anything available
        ]
        for level, attempt in enumerate(attempts):
            results, columns = self.query_properties(exclude_ids=exclude_ids, **attempt)
            if results:
                relaxed_fields = [
                    field for field in ("property_type", "location", "max_price")
                    if original.get(field) and not attempt.get("prop_type" if field == "property_type" else field)
                ]
                logger.info(
                    "Property search resolved at fallback level %d (relaxed=%s) - %d result(s) for "
                    "type=%r location=%r max_price=%r",
                    level, relaxed_fields, len(results), prop_type, location, max_price
                )
                return results, columns, (level == 0), relaxed_fields
        logger.info(
            "Property search found NOTHING at any fallback level for type=%r location=%r max_price=%r "
            "(excluding %d already-shown id(s)).",
            prop_type, location, max_price, len(exclude_ids or [])
        )
        return [], [], False, []

    _ORDINAL_WORDS = {
        "first": 0, "1st": 0, "pehla": 0, "pehli": 0,
        "second": 1, "2nd": 1, "dusra": 1, "dusri": 1,
        "third": 2, "3rd": 2, "teesra": 2, "teesri": 2,
    }

    def _resolve_selected_property(self, shown_properties, user_message):
        """
        Figures out exactly WHICH of the shown properties the user means, so
        we only ever save that one specific property as the lead - never the
        whole batch. Returns (selected_property_dict_or_None, ambiguous: bool).
        ambiguous=True means we genuinely can't tell and must ask, rather than
        guessing and saving the wrong property.
        """
        if not shown_properties:
            return None, False
        if len(shown_properties) == 1:
            return shown_properties[0], False

        lowered = user_message.lower()

        # 1. Ordinal words ("the second one", "pehla wala")
        for word, idx in self._ORDINAL_WORDS.items():
            if word in lowered and idx < len(shown_properties):
                return shown_properties[idx], False

        # 2. A bare digit 1/2/3 referring to position
        digit_match = re.search(r'(?<!\d)([1-9])(?!\d)', lowered)
        if digit_match:
            idx = int(digit_match.group(1)) - 1
            if 0 <= idx < len(shown_properties):
                return shown_properties[idx], False

        # 3. Property name or location mentioned explicitly
        matches = []
        for p in shown_properties:
            name = (p.get('property_name') or '').lower()
            loc = (p.get('location') or '').lower()
            loc_first_part = loc.split(',')[0].strip()
            if name and name in lowered:
                matches.append(p)
            elif loc_first_part and len(loc_first_part) > 3 and loc_first_part in lowered:
                matches.append(p)
        if len(matches) == 1:
            return matches[0], False

        # Genuinely ambiguous - multiple properties were shown and nothing in
        # the message tells us which one they mean.
        return None, True

    def save_to_csv(self, data):
        headers = ['timestamp', 'name', 'phone', 'property_type', 'property_name', 'location', 'price', 'area', 'face', 'outside_view', 'age', 'description', 'images']
        try:
            with _CSV_WRITE_LOCK:
                file_exists = os.path.isfile(self.user_listings_path)
                with open(self.user_listings_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore', restval='')
                    if not file_exists:
                        writer.writeheader()
                    data = dict(data)
                    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data['name'] = data.get('user_name', '')
                    data['phone'] = data.get('user_phone', '')
                    writer.writerow(data)
        except OSError as e:
            logger.error("Failed to write seller listing to CSV: %s", e, exc_info=True)
            raise

    def save_lead(self, search_criteria, selected_property, contact=None):
        """
        Records a buyer's confirmed interest in ONE specific property (never
        the whole batch that was shown) so a human sales rep can follow up on
        exactly what the person wants - this is the handoff point that keeps
        the lead from wandering off to another builder/dealer instead of
        being contacted.
        """
        contact = contact or {}
        selected_property = selected_property or {}
        headers = [
            'timestamp', 'name', 'phone', 'property_type', 'search_location', 'max_price',
            'chosen_property_name', 'chosen_property_location', 'chosen_property_price', 'chosen_property_link'
        ]
        try:
            with _CSV_WRITE_LOCK:
                file_exists = os.path.isfile(self.leads_path)
                with open(self.leads_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore', restval='')
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'name': contact.get('user_name', ''),
                        'phone': contact.get('user_phone', ''),
                        'property_type': search_criteria.get('property_type', ''),
                        'search_location': search_criteria.get('location', ''),
                        'max_price': search_criteria.get('max_price', ''),
                        'chosen_property_name': selected_property.get('property_name', ''),
                        'chosen_property_location': selected_property.get('location', ''),
                        'chosen_property_price': selected_property.get('total_price', ''),
                        'chosen_property_link': selected_property.get('property_link', ''),
                    })
        except OSError as e:
            logger.error("Failed to write lead to CSV: %s", e, exc_info=True)
            raise

    # ---------- main entry point ----------

    def handle_message(self, user_message, session_data=None):
        session_data = dict(session_data or {})

        # Guard against empty/whitespace-only messages (e.g. a WhatsApp
        # "reaction" or a stray webhook payload) so we never send an empty
        # extraction request to the LLM.
        if not user_message or not user_message.strip():
            lang = session_data.get("_lang", "hinglish")
            return {"text": TEXTS["empty_message"][lang], "media": None, "session_data": session_data}

        # Guard against abusively long input - protects LLM cost and avoids
        # sending unbounded text into the prompt.
        user_message = user_message.strip()[:MAX_MESSAGE_LENGTH]

        active_flow = session_data.get("_flow")             # 'buy' | 'buy_results' | 'sell' | None
        pending_field = session_data.get("_pending_field")  # the field we last asked for
        prior_lang = session_data.get("_lang")               # language locked in from earlier turns
        fail_count = session_data.get("_fail_count", 0)

        # 1. Extraction
        # Tell the LLM about the current conversational state so short replies like
        # "40 lakhs" or "show other option" are classified correctly instead of
        # being treated as a brand-new, context-less message (which used to reset
        # the whole flow back to the generic intro).
        flow_context = ""
        if active_flow in ("buy", "sell") and pending_field:
            flow_context = (
                f"\nIMPORTANT CONTEXT: We are already in the middle of a '{active_flow}' flow. "
                f"The last question asked the user was about '{pending_field}'. "
                f"If the user's message looks like an answer to that (a price, a place name, a property type, "
                f"a short phrase, a number, etc.), classify intent as 'continue_{active_flow}' and put that value "
                f"in params under the key '{pending_field}'. Only classify as 'greeting' or 'inquiry' if the "
                f"message clearly changes topic (e.g. an explicit hello, or an explicit new question about what "
                f"listings exist)."
            )
        elif active_flow == "contact" and pending_field:
            flow_context = (
                f"\nIMPORTANT CONTEXT: We just asked the user for their '{pending_field}' "
                f"({'their full name' if pending_field == 'user_name' else 'their phone number'}) before continuing "
                f"with their request. Their reply is almost certainly that value - classify intent as "
                f"'continue_contact' and put their reply in params under the key '{pending_field}'. Only classify "
                f"as 'greeting' if it's an unrelated fresh greeting."
            )
        elif active_flow == "buy_results":
            flow_context = (
                "\nIMPORTANT CONTEXT: We just showed the user some property listings. Distinguish carefully "
                "between two different things the user might mean:\n"
                "1) They want to CONFIRM/CHOOSE one of the properties just shown - e.g. 'I choose this one', "
                "'ye wala theek hai', 'book this one', 'I'm interested in this', 'yes this property', 'final, "
                "this one is good'. => classify intent as 'select_property' with empty params.\n"
                "2) They want to see MORE or DIFFERENT properties, or are giving new search criteria - e.g. "
                "'show other option', 'kuch aur dikhao', 'other properties', 'next', or a new location/budget/type. "
                "=> classify intent as 'continue_buy', with any new criteria in params.\n"
                "Only classify as 'greeting' if it's literally a greeting, or 'sell'/'continue_sell' if they "
                "clearly want to sell a property instead. When in doubt between 'select_property' and "
                "'continue_buy', a short affirmative/decisive reply about 'this one' or 'this property' means "
                "'select_property'; a request for alternatives or new criteria means 'continue_buy'."
            )

        # Heuristic safety net: some short decisive replies ("this one", "book this",
        # "ye wala") are easy for the LLM to misread as wanting more options instead
        # of confirming a choice - catch the common phrasing directly so a
        # confirmed buyer is never accidentally shown yet another property.
        selection_phrases = (
            "this one", "this property", "i choose", "i'll take", "i will take", "book this",
            "interested in this", "go with this", "finalize this", "ye wala", "ye property",
            "yehi", "isko book", "iske liye", "final kar", "ye le lo", "ye theek hai",
        )
        forced_selection = False
        if active_flow == "buy_results":
            lowered = user_message.lower()
            if any(phrase in lowered for phrase in selection_phrases):
                forced_selection = True

        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional Real Estate Dealer. Determine if the user wants to BUY, SELL, is just GREETING, or INQUIRING about what you have. \nIf providing details for a buy/sell flow, intent is 'continue_buy' or 'continue_sell'. Also detect whether the user is writing in plain English or in Hindi/Hinglish."
             + (
                 "\nIMPORTANT: Regardless of which intent you pick, ALWAYS extract into params any of "
                 "property_type, location, or max_price/budget that the user mentions in this message "
                 "(e.g. 'I have 15 lakh, suggest me a property' should still have max_price=15 lakh in params "
                 "even though it also sounds like a general request). Never drop a mentioned budget, location, "
                 "or property type just because the overall intent looks like a general inquiry."
                 if active_flow != "contact" else
                 "\nIMPORTANT: We are currently only collecting the user's name or phone number (see context "
                 "below). Do NOT extract property_type, location, or max_price from this message even if a "
                 "number is present - a phone number is NOT a budget. Only extract user_name/user_phone here."
             )
             + flow_context + "\n{format_instructions}"),
            ("user", "{user_message}")
        ])
        extraction_chain = extract_prompt | self.llm | self.parser

        extraction = None
        last_error = None
        for attempt in range(2):  # 1 retry on transient failure (timeout, rate limit, etc.)
            try:
                result = extraction_chain.invoke({"user_message": user_message, "format_instructions": self.parser.get_format_instructions()})
                if not isinstance(result, dict) or "intent" not in result:
                    raise ValueError(f"Extraction returned an unexpected shape: {result!r}")
                extraction = result
                break
            except Exception as e:
                last_error = e
                if attempt == 0:
                    time.sleep(0.5)

        if extraction is None:
            logger.warning("AI extraction failed after retry: %s", last_error, exc_info=True)
            lang = prior_lang or "hinglish"
            fail_count += 1
            session_data["_fail_count"] = fail_count
            if fail_count >= MAX_CONSECUTIVE_FAILURES:
                text = TEXTS["escalate"][lang]
                if SUPPORT_CONTACT:
                    text += f" 📞 {SUPPORT_CONTACT}"
                # Reset the failure counter but keep the rest of the session so
                # a human picking this up still has the prior context.
                session_data["_fail_count"] = 0
                return {"text": text, "media": None, "session_data": session_data}
            return {"text": TEXTS["extraction_failed"][lang], "media": None, "session_data": session_data}

        # Successful extraction - clear any accumulated failure count.
        session_data["_fail_count"] = 0

        intent = extraction.get("intent")
        # CRITICAL: the LLM's "intent" is just a free-form string (not an
        # enum), so it can come back as "Buy", "BUY", " buy ", "Continue_Buy",
        # etc. Every check in this file does an EXACT string match against
        # lowercase values like "buy" - without this normalization, any casing
        # variance silently fails every comparison (including the contact-info
        # gate), and the message falls through to a generic fallback instead
        # of doing what it's supposed to. This one line is what guarantees the
        # gate actually fires every time.
        if isinstance(intent, str):
            intent = intent.strip().lower().replace(" ", "_").replace("-", "_")
        VALID_INTENTS = {
            "buy", "sell", "continue_buy", "continue_sell", "select_property",
            "greeting", "inquiry", "continue_contact"
        }
        if intent not in VALID_INTENTS:
            logger.warning("LLM returned an unrecognized intent %r - treating as 'inquiry'.", intent)
            intent = "inquiry"

        params = extraction.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        # Lock in the detected language for this turn, falling back to whatever
        # was detected earlier in the conversation, then to Hinglish by default.
        detected_lang = self._normalize_language(extraction.get("language"))
        lang = detected_lang or prior_lang or "hinglish"

        current_data = dict(session_data)
        current_data["_lang"] = lang

        if active_flow == "contact":
            # Belt-and-braces: no matter what the LLM returns here, a phone
            # number or name can never be misread as a property_type/location/
            # max_price and silently satisfy those fields. This is what
            # actually prevents the buy/sell flow from being skipped.
            params = {k: v for k, v in params.items() if k in ("user_name", "user_phone")}

        # Only merge in fields the LLM actually gave a real value for, so a
        # missing/None field in this turn can't wipe out data from an earlier turn.
        for key, value in params.items():
            if value not in (None, '', 'unknown', 'Unknown'):
                current_data[key] = value

        # Safety net #1: mid buy/sell/contact question-answer flow.
        if active_flow in ("buy", "sell", "contact") and pending_field and intent in ("greeting", "inquiry", None):
            if not current_data.get(pending_field) and user_message:
                current_data[pending_field] = user_message
            intent = f"continue_{active_flow}"

        # If we're specifically waiting on a phone number, validate it before
        # accepting - catches the LLM (or the safety net above) grabbing junk
        # text as a "phone number" that a human would never actually reach.
        phone_was_invalid = False
        if active_flow == "contact" and pending_field == "user_phone" and intent == "continue_contact":
            if not self._is_valid_phone(current_data.get("user_phone")):
                current_data.pop("user_phone", None)
                phone_was_invalid = True

        # Safety net #2: right after showing results, ambiguous/failed
        # classifications ('inquiry' or None) for a "show me more" style reply
        # must not be misread as something unrelated. A genuine, clearly-detected
        # 'greeting' is intentionally EXCLUDED here - if the user actually says
        # "hi"/"hello", they get a real greeting and a clean reset, not another
        # forced property search. (Forcing greeting here was the bug that made
        # a stale buy_results session hijack every future "hello".)
        if active_flow == "buy_results" and intent in ("inquiry", None):
            intent = "continue_buy"

        # Safety net #3: decisive "this one" / "book this" style replies must always
        # close the sale, never be reinterpreted as a request for more options.
        if forced_selection:
            intent = "select_property"

        # An 'inquiry' ("what do you have?") is a live opportunity, not a dead end -
        # fold it straight into the buying funnel so the flow state (_flow /
        # _pending_field) actually gets tracked. Without this, every follow-up
        # reply arrives with zero memory of the conversation and keeps landing
        # back on the same generic "what type of property?" message forever.
        came_from_pure_inquiry = (intent == "inquiry" and not active_flow)
        if intent == "inquiry":
            intent = "continue_buy"

        # Persona Prompt - matches the user's own language so the agent doesn't
        # answer in Hinglish when the user is clearly writing in English, or vice versa.
        if lang == "english":
            persona_prompt = "You are a friendly, professional Real Estate Dealer. Speak in plain, natural English like a real human. Be helpful, direct, and sales-oriented. Don't be robotic."
        else:
            persona_prompt = "You are a friendly, professional Real Estate Dealer. Speak in Hinglish (a mix of Hindi and English) like a real human. Be helpful, direct, and sales-oriented. Don't be robotic. Don't use 'unknown' words."

        logger.info(
            "Routing message (len=%d chars): intent=%s lang=%s active_flow=%s",
            len(user_message), intent, lang, active_flow
        )

        try:
            return self._route_intent(intent, current_data, user_message, lang, persona_prompt, came_from_pure_inquiry, phone_was_invalid)
        except Exception as e:
            # Last-resort safety net: whatever goes wrong inside the actual
            # business logic (DB hiccup, bad data shape, etc.), the user
            # always gets a graceful reply instead of a broken webhook call.
            logger.error("Unhandled error while routing intent '%s': %s", intent, e, exc_info=True)
            return {"text": TEXTS["unexpected_error"][lang], "media": None, "session_data": current_data}

    # Intents that must not proceed until we actually have the user's name and
    # a valid phone number - this is what guarantees every lead/listing saved
    # afterward has real contact details attached.
    CONTACT_GATED_INTENTS = ("buy", "continue_buy", "sell", "continue_sell", "select_property", "continue_contact")

    def _route_intent(self, intent, current_data, user_message, lang, persona_prompt, came_from_pure_inquiry, phone_was_invalid=False):
        # 1. Greetings - no contact info required just to say hello.
        if intent == "greeting":
            return {
                "text": TEXTS["greeting"][lang],
                "media": None,
                "session_data": self._end_conversation(current_data)
            }

        # 1b. Contact capture gate - runs before buy/sell/selection can proceed.
        if intent in self.CONTACT_GATED_INTENTS:
            if intent != "continue_contact":
                # Remember what the user was actually trying to do, so once
                # contact info is collected we resume exactly that, instead of
                # losing their original request.
                current_data["_resume_intent"] = intent
            resume_intent = current_data.get("_resume_intent") or "buy"

            name_ok = bool(current_data.get("user_name"))
            phone_ok = self._is_valid_phone(current_data.get("user_phone"))

            if not (name_ok and phone_ok):
                current_data["_flow"] = "contact"
                if not name_ok:
                    current_data["_pending_field"] = "user_name"
                    text = REQUIRED_CONTACT["user_name"][lang]
                else:
                    current_data["_pending_field"] = "user_phone"
                    text = REQUIRED_CONTACT["user_phone_invalid"][lang] if phone_was_invalid else REQUIRED_CONTACT["user_phone"][lang]
                return {"text": text, "media": None, "session_data": current_data}

            # Contact info complete - drop the sub-flow markers and resume
            # whatever the user originally asked for.
            current_data.pop("_flow", None)
            current_data.pop("_pending_field", None)
            current_data.pop("_resume_intent", None)
            intent = resume_intent

        # 2. Buying Flow (Interactive) - also handles what used to be the
        # separate 'inquiry' intent, since a browsing question is really just
        # the start of a buying conversation.
        if intent in ["buy", "continue_buy"]:
            missing = [k for k in REQUIRED_BUY if not current_data.get(k)]

            if missing:
                next_field = missing[0]
                current_data["_flow"] = "buy"
                current_data["_pending_field"] = next_field
                # Keep the friendlier "what do we have" framing the first time
                # someone asks a general question, then fall back to the
                # direct field-specific question for every turn after that.
                if came_from_pure_inquiry and next_field == "property_type":
                    text = TEXTS["inquiry"][lang]
                else:
                    text = REQUIRED_BUY[next_field][lang]
                return {"text": text, "media": None, "session_data": current_data}

            search_key = {
                "property_type": None if self._is_wildcard(current_data.get("property_type")) else current_data.get("property_type"),
                "location": None if self._is_wildcard(current_data.get("location")) else current_data.get("location"),
                "max_price": current_data.get("max_price"),
            }
            # Only keep paginating through the same result set if the search
            # criteria actually stayed the same; a new location/budget/type
            # means it's a fresh search, so we shouldn't exclude old matches.
            if current_data.get("_last_search") == search_key:
                shown_ids = current_data.get("_shown_ids", [])
            else:
                shown_ids = []

            properties, columns, exact_match, relaxed_fields = self.find_best_properties(
                prop_type=search_key["property_type"],
                location=search_key["location"],
                max_price=search_key["max_price"],
                exclude_ids=shown_ids,
            )

            current_data.pop("_pending_field", None)
            current_data["_last_search"] = search_key

            if not properties:
                if shown_ids:
                    # We'd already shown everything that matched; nothing new to add.
                    current_data["_flow"] = "buy_results"
                    return {"text": TEXTS["exhausted"][lang], "media": None, "session_data": current_data}
                # Genuinely nothing in the entire database - only now do we apologize.
                # This message is deterministic (not LLM-generated) so a downed
                # LLM API can never block us from replying here.
                return {"text": TEXTS["no_listings_at_all"][lang], "media": None, "session_data": self._end_conversation(current_data)}

            # Build deterministic (non-hallucinated) property listings straight
            # from the database rows.
            property_dicts = [self._row_to_dict(row, columns) for row in properties]
            listing_blocks = [self._build_property_caption(p, lang) for p in property_dicts]

            # Pick the most specific, honest framing for *why* this isn't an exact
            # match, so the user is actively steered toward a real alternative
            # (a different location, or a slightly different budget) instead of
            # getting a generic "closest options" brush-off.
            if exact_match:
                intro = TEXTS["exact_intro"][lang]
            elif relaxed_fields == ["location"]:
                intro = TEXTS["fallback_intro_location"][lang]
                logger.info("No properties in requested location %r - steering user to other locations.", search_key["location"])
            elif relaxed_fields == ["max_price"]:
                intro = TEXTS["fallback_intro_budget"][lang]
            else:
                intro = TEXTS["fallback_intro"][lang]
            text = intro + "\n\n" + "\n\n".join(listing_blocks) + "\n\n" + TEXTS["more_options_cta"][lang]

            new_ids = [p.get("id") for p in property_dicts if p.get("id") is not None]
            current_data["_shown_ids"] = list(shown_ids) + new_ids
            current_data["_flow"] = "buy_results"
            # Lightweight record (not the full rows) purely so that if the user
            # says "I choose this one" next, we can log exactly what was shown
            # for the sales team to follow up on.
            current_data["_last_shown_properties"] = [
                {
                    "property_name": p.get("property_name"),
                    "location": p.get("location"),
                    "total_price": p.get("total_price"),
                    "property_link": p.get("property_link"),
                }
                for p in property_dicts
            ]

            media_url = property_dicts[0].get("property_img_link") or None
            # Individual per-property media so the caller (e.g. WhatsApp sender) can
            # send each property's own image with its own accurate caption.
            properties_payload = [
                {"image": p.get("property_img_link") or None, "caption": block}
                for p, block in zip(property_dicts, listing_blocks)
            ]

            return {
                "text": text,
                "media": media_url,
                "properties": properties_payload,
                "session_data": current_data,
            }

        # 3. Property Selection (user confirms/chooses one of the shown properties)
        # This is the closing moment - the user is satisfied and ready to move
        # forward, so we reassure them, log the lead for a human follow-up, and
        # deliberately STOP showing more properties so they aren't pushed toward
        # second-guessing their choice or wandering off to another dealer.
        if intent == "select_property":
            shown_properties = current_data.get("_last_shown_properties", [])
            selected, ambiguous = self._resolve_selected_property(shown_properties, user_message)

            if selected is None and ambiguous:
                # Multiple properties were shown and nothing tells us which one
                # they mean - ask instead of guessing (and definitely instead
                # of saving the whole batch as "the" chosen property).
                current_data["_flow"] = "buy_results"
                numbered = "\n".join(
                    f"{i + 1}. {p.get('property_name') or 'Property'} - {p.get('location', '')}"
                    for i, p in enumerate(shown_properties)
                )
                text = TEXTS["clarify_selection"][lang] + "\n\n" + numbered
                return {"text": text, "media": None, "session_data": current_data}

            search_criteria = {
                "property_type": current_data.get("property_type", ""),
                "location": current_data.get("location", ""),
                "max_price": current_data.get("max_price", ""),
            }
            contact = {
                "user_name": current_data.get("user_name", ""),
                "user_phone": current_data.get("user_phone", ""),
            }
            try:
                self.save_lead(search_criteria, selected, contact=contact)
                logger.info(
                    "Lead captured: name=%s phone=%s criteria=%s, chosen property=%s",
                    contact["user_name"], contact["user_phone"], search_criteria,
                    (selected or {}).get("property_name")
                )
            except Exception:
                # Even if the lead couldn't be logged (disk issue, permissions),
                # the user must still get their reassuring closing message -
                # losing the write is bad, silently losing the *user* is worse.
                logger.error("Lead capture failed for criteria=%s - closing message still sent.", search_criteria, exc_info=True)
            return {"text": TEXTS["selection_confirmed"][lang], "media": None, "session_data": self._end_conversation(current_data)}

        # 4. Selling Flow (Interactive)
        if intent in ["sell", "continue_sell"]:
            missing = [k for k in REQUIRED_SELL if not current_data.get(k)]

            if not missing:
                current_data.pop("_flow", None)
                current_data.pop("_pending_field", None)
                try:
                    self.save_to_csv(current_data)
                    logger.info(
                        "Seller listing captured: type=%s location=%s price=%s",
                        current_data.get("property_type"), current_data.get("location"), current_data.get("price")
                    )
                except Exception:
                    logger.error("Could not persist seller listing - notifying user anyway.", exc_info=True)
                return {"text": TEXTS["sell_done"][lang], "media": None, "session_data": self._end_conversation(current_data)}

            next_field = missing[0]
            current_data["_flow"] = "sell"
            current_data["_pending_field"] = next_field
            return {"text": REQUIRED_SELL[next_field][lang], "media": None, "session_data": current_data}

        return {"text": TEXTS["fallback"][lang], "media": None, "session_data": current_data}


if __name__ == "__main__":
    agent = RealEstateAgent()
    # Test seller flow
    res = agent.handle_message("I want to sell my flat")
    print(res['text'])
    res2 = agent.handle_message("It's a luxury flat in Mumbai", session_data=res.get('session_data'))
    print(res2['text'])
