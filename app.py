import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from agent.agent import RealEstateAgent

load_dotenv()

logger = logging.getLogger("whatsapp_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

# ---------- startup configuration & validation ----------

ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")  # used to verify webhook signatures
API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v20.0")

_missing = [name for name, val in [
    ("WHATSAPP_ACCESS_TOKEN", ACCESS_TOKEN),
    ("WHATSAPP_PHONE_NUMBER_ID", PHONE_NUMBER_ID),
    ("WHATSAPP_VERIFY_TOKEN", VERIFY_TOKEN),
] if not val]
if _missing:
    raise RuntimeError(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        "Set these before starting the app (see .env.example)."
    )
if not APP_SECRET:
    logger.warning(
        "WHATSAPP_APP_SECRET is not set - incoming webhook requests will NOT be "
        "signature-verified. Anyone who discovers your webhook URL could POST fake "
        "messages. Set WHATSAPP_APP_SECRET in production."
    )

WHATSAPP_API_URL = f"https://graph.facebook.com/{API_VERSION}/{PHONE_NUMBER_ID}/messages"
WHATSAPP_TEXT_LIMIT = 4096  # Meta's max length for a text message body

app = Flask(__name__)
agent = RealEstateAgent()  # fails fast at startup if LLM credentials are missing

# ---------- persistent state (SQLite - survives restarts & is shared across workers) ----------

DATABASE_DIR = os.getenv("DATABASE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
APP_STATE_DB = os.path.join(DATABASE_DIR, "app_state.db")
DEDUP_WINDOW = timedelta(hours=24)  # how long we remember a processed message id


def _get_state_conn():
    conn = sqlite3.connect(APP_STATE_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _init_state_db():
    conn = _get_state_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                phone TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


_init_state_db()


def get_session(phone):
    try:
        conn = _get_state_conn()
        try:
            row = conn.execute("SELECT data FROM sessions WHERE phone = ?", (phone,)).fetchone()
            return json.loads(row[0]) if row else {}
        finally:
            conn.close()
    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error("Failed to load session for %s: %s", phone, e, exc_info=True)
        return {}  # degrade gracefully - worst case we ask the user a question again


def save_session(phone, data):
    try:
        conn = _get_state_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (phone, data, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(phone) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
                (phone, json.dumps(data), datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.error("Failed to save session for %s: %s", phone, e, exc_info=True)


def clear_session(phone):
    try:
        conn = _get_state_conn()
        try:
            conn.execute("DELETE FROM sessions WHERE phone = ?", (phone,))
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.error("Failed to clear session for %s: %s", phone, e, exc_info=True)


def already_processed(message_id):
    """True if we've already handled this WhatsApp message id - Meta retries
    webhook deliveries on timeouts, so without this the same message could
    be answered twice."""
    if not message_id:
        return False
    try:
        conn = _get_state_conn()
        try:
            cutoff = (datetime.utcnow() - DEDUP_WINDOW).isoformat()
            conn.execute("DELETE FROM processed_messages WHERE processed_at < ?", (cutoff,))
            row = conn.execute("SELECT 1 FROM processed_messages WHERE message_id = ?", (message_id,)).fetchone()
            if row:
                return True
            conn.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id, processed_at) VALUES (?, ?)",
                (message_id, datetime.utcnow().isoformat())
            )
            conn.commit()
            return False
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.error("Dedup check failed for message %s: %s", message_id, e, exc_info=True)
        return False  # if in doubt, still process the message rather than silently drop it


# ---------- WhatsApp send helpers ----------

def _post_to_whatsapp(payload, retries=1):
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    last_response = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(WHATSAPP_API_URL, headers=headers, json=payload, timeout=15)
            last_response = response
            if response.status_code < 500:
                # 2xx = success, 4xx = won't be fixed by retrying (bad token, bad payload)
                break
        except requests.RequestException as e:
            logger.error("WhatsApp API request failed (attempt %d): %s", attempt + 1, e, exc_info=True)
            if attempt < retries:
                time.sleep(1)
            continue
        if attempt < retries:
            time.sleep(1)

    if last_response is None:
        logger.error("WhatsApp API call failed with no response after %d attempt(s).", retries + 1)
        return None

    logger.info("WhatsApp API status: %s", last_response.status_code)
    if last_response.status_code >= 400:
        logger.error("WhatsApp API error response: %s", last_response.text)
    try:
        return last_response.json()
    except ValueError:
        return None


def _chunk_text(text, limit=WHATSAPP_TEXT_LIMIT):
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


def _is_valid_media_url(url):
    return isinstance(url, str) and url.startswith("https://")


def send_whatsapp_message(to, text, media_url=None):
    if media_url and not _is_valid_media_url(media_url):
        logger.warning("Ignoring non-HTTPS/invalid media URL %r - sending as text instead.", media_url)
        media_url = None

    if media_url:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": {"link": media_url, "caption": text[:1024]}  # WhatsApp caption limit
        }
        result = _post_to_whatsapp(payload)
        if result is not None and "error" not in result:
            return result
        # Image send failed (e.g. Meta couldn't fetch that URL) - don't lose the
        # message, fall back to sending the same content as plain text.
        logger.warning("Image send failed for %s, falling back to text.", to)

    results = []
    for chunk in _chunk_text(text):
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": chunk}
        }
        results.append(_post_to_whatsapp(payload))
    return results


def deliver_agent_reply(to, result):
    """
    Sends the agent's reply to WhatsApp. If the agent returned individual
    per-property listings (each with its own image + accurate caption), each
    one is sent as its own image message so the user actually sees every
    property's photo and details, instead of only the first one.
    """
    properties = result.get("properties")

    if properties:
        send_whatsapp_message(to, result["text"])
        for prop in properties:
            image = prop.get("image")
            caption = prop.get("caption", "")
            send_whatsapp_message(to, caption, media_url=image)
    else:
        send_whatsapp_message(to, result.get("text", ""), result.get("media"))


# ---------- webhook signature verification ----------

def _verify_signature(raw_body, signature_header):
    if not APP_SECRET:
        return True  # verification disabled (already warned about this at startup)
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.split("sha256=", 1)[1]
    return hmac.compare_digest(expected, provided)


# ---------- routes ----------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/whatsapp", methods=['GET', 'POST'])
def whatsapp_webhook():
    # 1. Handle Webhook Verification (GET)
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        if mode == 'subscribe' and token == VERIFY_TOKEN:
            return challenge, 200
        return 'Verification failed', 403

    # 2. Handle Incoming Messages (POST)
    if not _verify_signature(request.get_data(), request.headers.get("X-Hub-Signature-256")):
        logger.warning("Rejected webhook POST with invalid signature.")
        return jsonify({"status": "invalid signature"}), 401

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception as e:
        logger.error("Failed to parse webhook JSON: %s", e, exc_info=True)
        return jsonify({"status": "bad request"}), 400

    if data.get("object") != "whatsapp_business_account":
        # Not a message event we care about (could be a different subscribed field) - ack and ignore.
        return jsonify({"status": "ignored"}), 200

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                try:
                    _handle_single_message(msg, lang_hint=value)
                except Exception as e:
                    # A single malformed/unexpected message must never take down
                    # the whole webhook handler or block the rest of the batch.
                    logger.error("Unhandled error processing message %s: %s", msg.get("id"), e, exc_info=True)

    return jsonify({"status": "ok"}), 200


def _handle_single_message(msg, lang_hint=None):
    message_id = msg.get("id")
    from_number = msg.get("from")

    if not from_number:
        logger.warning("Message with no 'from' number, skipping: %r", msg)
        return

    if already_processed(message_id):
        logger.info("Skipping duplicate/already-processed message %s from %s.", message_id, from_number)
        return

    msg_type = msg.get("type")
    if msg_type != "text":
        # We only understand text today - reply with a friendly nudge instead
        # of silently dropping images/audio/documents/interactive replies.
        friendly_note = (
            "Ji, abhi main sirf text messages samajh sakta hoon. Kripya apni requirement "
            "text mein likh dijiye. 🙂"
        )
        send_whatsapp_message(from_number, friendly_note)
        return

    text_body = (msg.get("text") or {}).get("body", "")
    if not text_body.strip():
        return

    session_data = get_session(from_number)

    try:
        result = agent.handle_message(text_body, session_data=session_data)
    except Exception as e:
        logger.error("Agent failed to handle message from %s: %s", from_number, e, exc_info=True)
        send_whatsapp_message(
            from_number,
            "Sorry, kuch technical dikkat aa gayi. Kripya thodi der baad dobara try karein."
        )
        return

    if result.get("session_clear"):
        clear_session(from_number)
    elif result.get("session_data") is not None:
        save_session(from_number, result["session_data"])

    deliver_agent_reply(from_number, result)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    if debug_mode:
        logger.warning("Starting in DEBUG mode - never do this in production (remote code execution risk).")
    # For real deployment, run this behind gunicorn/uwsgi instead of the Flask
    # dev server, e.g.: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(port=int(os.getenv("PORT", 5000)), debug=debug_mode)