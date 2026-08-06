# Real Estate WhatsApp AI Agent

An AI-powered real estate assistant that talks to buyers and sellers directly
on **WhatsApp**. It understands both English and Hinglish, helps buyers
search a live property database and books their interest as a qualified
lead, helps sellers submit new listings, and hands every serious lead to
your sales team with a name and phone number attached — so nobody falls
through the cracks.

Built for a real estate builder/dealer to deploy on their own WhatsApp
Business number, backed by Meta's WhatsApp Cloud API and an LLM (Groq's
Llama 3.3, with OpenAI as a fallback) for natural-language understanding.

---

## What it actually does

- **Buyers**: say what they want in plain language ("mujhe Jagatpura mein
  40 lakh ka flat chahiye" / "I want a 3BHK villa under 60 lakh") — the
  agent asks anything missing (type, location, budget), searches the
  database, and shows real listings with photos, price, area, and a link.
- **No dead ends**: if there's no exact match, it automatically relaxes the
  search (budget → location → type) and shows the closest available
  options instead of saying "nothing found."
- **Closes the loop**: once a buyer says "I'll take this one," the agent
  confirms which exact property they mean (asking for clarification if more
  than one was shown), captures their name and phone number, and saves it
  as a lead — so your team can call them back.
- **Sellers**: can list a property for sale by answering a short series of
  questions (type, location, price, area, etc.) — saved to a CSV for your
  team to review.
- **Sold-out properties are never shown** — every search filters on
  `status = 'available'`; a simple CLI script lets you mark a unit sold the
  moment it's off the market.
- **Speaks the user's language** — replies in English if they wrote in
  English, Hinglish if they wrote in Hinglish, and remembers this for the
  rest of the conversation.
- **Production-hardened**: persistent sessions (survive restarts and work
  across multiple server workers), webhook signature verification, retry
  logic around every AI/network call, structured logging, and graceful
  fallbacks everywhere instead of crashes.

---

## Project structure

```
.
├── app.py                     # Flask webhook - talks to WhatsApp
├── agent/
│   └── agent.py                # The AI agent - all conversation logic lives here
├── setup_db.py                 # One-time (or re-run) script: builds properties.db from your CSVs
├── mark_property_status.py     # CLI to mark a property sold / available again
├── requirements.txt            # Pinned Python dependencies
├── .env.example                # Template for your real .env (never commit the real one)
└── database/
    ├── plots_dataset.csv           # YOUR source data (you provide these)
    ├── flats_dataset.csv           # YOUR source data (you provide these)
    ├── house_and_villa_dataset.csv # YOUR source data (you provide these)
    ├── properties.db               # Built by setup_db.py - the live property listings
    ├── app_state.db                # Auto-created - WhatsApp sessions + message dedup
    ├── user_listings.csv           # Auto-created - properties submitted by sellers
    └── interested_leads.csv        # Auto-created - buyers who confirmed interest
```

### File-by-file

| File | Purpose |
|---|---|
| **`app.py`** | The Flask web server that receives WhatsApp messages via Meta's webhook, verifies they're genuine (signature check), de-duplicates retried deliveries, calls the agent, and sends the reply back — including sending each property's photo individually with its own caption. This is the only file that talks to the WhatsApp API. |
| **`agent/agent.py`** | The actual brain. Classifies what the user wants (buy / sell / greeting / choosing a property), tracks conversation state (what question was just asked, what's already been collected), runs the database search with automatic fallback, builds property descriptions **directly from the database** (never invented by the AI, to avoid wrong facts), and writes leads/listings to CSV. Everything here is plain Python — no WhatsApp-specific code. |
| **`setup_db.py`** | Run this once (and any time your source CSVs change) to build `database/properties.db`. Reads `plots_dataset.csv`, `flats_dataset.csv`, and `house_and_villa_dataset.csv`, normalizes prices/areas, and respects a sold/availability column in your source data if one exists. |
| **`mark_property_status.py`** | A command you or your team runs whenever a unit sells, e.g. `python mark_property_status.py --name "Green Residency" --status sold` — instantly removes it from every future buyer search. |
| **`requirements.txt`** | Exact versions of every Python package this project needs. |
| **`.env.example`** | A template listing every environment variable the app needs, with comments on where to get each value. Copy it to `.env` and fill in real values — `.env` itself should **never** be committed to git. |
| **`database/properties.db`** | The live SQLite database of properties buyers search against. Rebuilt by `setup_db.py`. |
| **`database/user_listings.csv`** | Every property a seller has submitted through the bot, with their name/phone attached. |
| **`database/interested_leads.csv`** | Every buyer who confirmed interest in a specific property, with their name, phone, search criteria, and exactly which property they chose. This is the file your sales team should be watching. |
| **`database/app_state.db`** | Internal bookkeeping — active WhatsApp conversation state per phone number, and a record of already-processed message IDs (so a WhatsApp retry never double-answers someone). You don't need to touch this. |

---

## How a conversation flows

```
User: "hi"
Agent: Greets, asks buy or sell.

User: "I want to buy a flat"
Agent: Asks for name → asks for phone number (only once per conversation)
Agent: Asks type / location / budget for whatever wasn't already mentioned.
Agent: Searches the database. Shows up to 3 matching properties (or the
       closest available alternatives, with an honest note about why).

User: "show other option"
Agent: Shows a different batch, remembering what's already been shown.

User: "I choose this one" / "the second one" / "Blue Heights"
Agent: Figures out exactly which property was meant (asking for
       clarification if genuinely ambiguous), saves it as a lead with the
       user's name and phone, and confirms the team will follow up.
```

Selling works the same way, just collecting different fields (property
name, price, area, facing, etc.) before saving to `user_listings.csv`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Then fill in `.env` with real values:

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `OPENAI_API_KEY` (fallback, optional if Groq is set) | platform.openai.com |
| `WHATSAPP_ACCESS_TOKEN` | Meta Business Settings → System Users → Generate Token (use a **permanent** System User token, not the 24-hour dashboard token) |
| `WHATSAPP_PHONE_NUMBER_ID` | Meta App Dashboard → WhatsApp → API Setup |
| `WHATSAPP_VERIFY_TOKEN` | Any string you make up — you'll enter the same string in Meta's webhook config |
| `WHATSAPP_APP_SECRET` | Meta App Dashboard → Settings → Basic → App Secret (enables webhook signature verification — strongly recommended) |
| `SUPPORT_CONTACT` | Optional — phone number shown to a user if the bot has to escalate to a human |

### 3. Add your property data

Place your CSV files in `database/`:
- `plots_dataset.csv`
- `flats_dataset.csv`
- `house_and_villa_dataset.csv`

Then build the database:

```bash
python setup_db.py
```

Re-run this any time your source CSVs are updated.

### 4. Run the app

**Local development:**
```bash
python app.py
```

**Production (recommended — do not use `python app.py` in production):**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 5. Connect it to WhatsApp

In the Meta App Dashboard, set your webhook URL to `https://<your-domain>/whatsapp`
and the verify token to whatever you set as `WHATSAPP_VERIFY_TOKEN`.

---

## Day-to-day operations

- **A property sells** → `python mark_property_status.py --name "<name>" --status sold`
- **Follow up on buyer leads** → check `database/interested_leads.csv`
- **Follow up on new seller listings** → check `database/user_listings.csv`
- **Refresh property data** → update your CSVs, re-run `python setup_db.py`
- **Check logs** → currently printed to console/stdout; set `LOG_LEVEL=DEBUG` in `.env` for more detail

---

## Notes on reliability

- Property facts shown to users (price, area, location, etc.) are always
  pulled directly from the database — the AI never invents or paraphrases
  numbers, to avoid giving a buyer wrong information.
- Every AI call has a retry and a safe fallback message if the AI provider
  is briefly unavailable; the bot never just goes silent.
- If the AI genuinely can't understand a user after a couple of tries, it
  automatically offers to connect them to a human instead of looping forever.