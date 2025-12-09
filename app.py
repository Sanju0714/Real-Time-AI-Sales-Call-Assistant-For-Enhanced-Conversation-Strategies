import streamlit as st
import os
import queue
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq
from textblob import TextBlob
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import google.generativeai as genai
import json
import io


# --- CONFIGURATION & CLIENT INITIALIZATION ---
load_dotenv()
st.set_page_config(layout="wide", page_title="AI Sales Co-Pilot")

# --- CORE SETTINGS ---
CHUNK_DURATION = 10
OVERLAP_DURATION = 1 #change 
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = int(0.05 * SAMPLE_RATE)
MAX_BLOCKS_PER_CHUNK = int(CHUNK_DURATION * (SAMPLE_RATE / BLOCKSIZE))

# --- API CLIENTS SETUP ---
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = genai.GenerationConfig(response_mime_type="application/json")
        gemini_client = genai.GenerativeModel('models/gemini-flash-latest', generation_config=generation_config)
        st.session_state.active_llm = "Gemini Pro"
    except Exception as e:
        st.warning(f"Failed to initialize Gemini client: {e}.")
        st.session_state.active_llm = "N/A"
else:
    st.session_state.active_llm = "N/A"

# <<< RE-ENGINEERED GOOGLE SHEET CONNECTION >>>
@st.cache_resource
def connect_and_get_sheets():
    """Connects to Google Sheets and fetches all required worksheets."""
    sheets = {
        "sales_log": None, "crm_customers": None,
        "crm_products": None, "crm_interactions": None
    }
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
        client = gspread.authorize(creds)

        # Connect to the Sales Log spreadsheet
        sales_log_conn = client.open("Sales_Call_Analysis")
        sheets["sales_log"] = sales_log_conn.sheet1

        # Connect to the single CRM Database spreadsheet
        crm_db_conn = client.open("CRM_Database")
        sheets["crm_customers"] = crm_db_conn.worksheet("Customers")
        sheets["crm_products"] = crm_db_conn.worksheet("Products")
        sheets["crm_interactions"] = crm_db_conn.worksheet("Interactions")
        
        return sheets, None
    except FileNotFoundError:
        return sheets, "`credentials.json` not found. Google Sheets integration disabled."
    except gspread.exceptions.SpreadsheetNotFound as e:
        return sheets, f"Spreadsheet not found: {e}. Please ensure 'Sales_Call_Analysis' and 'CRM_Database' exist."
    except gspread.exceptions.WorksheetNotFound as e:
         return sheets, f"A worksheet (tab) is missing from 'CRM_Database': {e}. Required tabs: 'Customers', 'Products', 'Interactions'."
    except Exception as e:
        return sheets, f"An unexpected error occurred connecting to Google Sheets: {e}"

# Fetch all sheets at startup
google_sheets, connection_error = connect_and_get_sheets()
if connection_error:
    st.error(connection_error)

sales_log_sheet = google_sheets["sales_log"]
crm_customers_sheet = google_sheets["crm_customers"]
crm_products_sheet = google_sheets["crm_products"]
crm_interactions_sheet = google_sheets["crm_interactions"]


# <<< RE-ENGINEERED CRM DATA FETCHER >>>
@st.cache_data(ttl=600)
def fetch_full_customer_profile(_customers_sheet, _products_sheet, _interactions_sheet, email):
    if not all([_customers_sheet, _products_sheet, _interactions_sheet]):
        return None, "One or more CRM Google Sheets are not connected."
    try:
        customers_data = _customers_sheet.get_all_records()
        customer_profile = next((c for c in customers_data if c.get('Contact_Email') == email), None)
        if not customer_profile:
            return None, f"Customer with email '{email}' not found."
        customer_id = customer_profile.get('Customer_ID')
        interactions_data = _interactions_sheet.get_all_records()
        customer_interactions = [i for i in interactions_data if i.get('Customer_ID') == customer_id]
        products_data = _products_sheet.get_all_records()
        # Create a dictionary for O(1) lookups instead of O(n) searches
        products_dict = {p.get('Product_ID'): p for p in products_data}

        full_interaction_history, purchase_history_summary = [], []
        for interaction in customer_interactions:
            product_id_string = interaction.get('Product_Purchased')
            if product_id_string and product_id_string != "(none)":
                product_ids = [pid.strip() for pid in str(product_id_string).split(',')]
                for product_id in product_ids:
                    product_details = products_dict.get(product_id, {})
                    if interaction.get('Outcome') == 'Sale Closed':
                        purchase_history_summary.append(product_details.get('Product_Name', 'Unknown Product'))
            full_interaction_history.append({
                "date": interaction.get('Date'), "product_id": interaction.get('Product_Purchased'),
                "outcome": interaction.get('Outcome'), "notes": interaction.get('Notes_Keywords')})
        customer_profile['interaction_history'] = full_interaction_history
        customer_profile['purchase_history_summary'] = ", ".join(sorted(list(set(purchase_history_summary)))) if purchase_history_summary else "None"
        return customer_profile, None
    except Exception as e:
        return None, f"Error fetching from CRM: {e}"

# --- CORE AI & DATA FUNCTIONS (No changes needed below this line) ---
@st.cache_data(ttl=300)
def get_total_calls(_sheet):
    if not _sheet: return 0
    try: return len(_sheet.get_all_records())
    except Exception: return 0

@st.cache_data(ttl=300)
def fetch_latest_history_from_sheet(_sheet):
    if not _sheet: return []
    try:
        all_data = _sheet.get_all_records()
        history_rows = all_data[-3:]
        history_list = []
        for row in reversed(history_rows):
            history_list.append({
                "timestamp": row.get("Timestamp", "N/A"), "transcript": row.get("Transcript", "N/A"),
                "sentiment": row.get("Sentiment", "N/A"), "summary": row.get("LLM Summary", "N/A")})
        return history_list
    except Exception: return []

def get_live_call_suggestions(customer_profile, conversation_history, latest_transcript):
    if not gemini_client: return {"error": "LLM client not configured."}
    prompt = f"""
    You are an expert AI Sales Co-Pilot. Your goal is to help a salesperson have a successful, personalized conversation by providing real-time analysis and suggestions.
    **CUSTOMER PROFILE (from CRM):**
    - Name: {customer_profile.get('Name', 'N/A')}
    - Purchase History: {customer_profile.get('purchase_history_summary', 'N/A')}
    - Past Interaction Notes: {customer_profile.get('interaction_history', [])}
    **LIVE CONVERSATION HISTORY (So Far):** {" ".join(conversation_history)}
    **LATEST UTTERANCE FROM CUSTOMER (10 seconds):** "{latest_transcript}"
    **YOUR TASK:** Return a JSON object with three keys:
    1. "sentiment": Analyze the sentiment of the "LATEST UTTERANCE". (Options: "Positive", "Negative", "Neutral", "Questioning").
    2. "product_recommendation": If the conversation suggests an opportunity to upsell or cross-sell, recommend a specific product and a brief reason. If not, this must be an empty string.
    3. "next_step_suggestion": Suggest the EXACT next question the salesperson should ask or a phrase to handle a potential objection. This must be actionable and conversational.
    """
    # ... inside get_live_call_suggestions
    try:
        response = gemini_client.generate_content(prompt)
        try:
            # Attempt to parse the JSON response
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Handle cases where the LLM output is not valid JSON
            return {"error": "AI returned an invalid format. Could not parse suggestions."}
    except Exception as e:
        return {"error": f"Gemini API Error: {e}"}

def perform_final_analysis_and_log(sheet_to_log, customer_profile, full_transcript):
    """Analyzes the full transcript and then logs it."""
    if not full_transcript.strip():
        st.session_state.final_summary = "No speech was detected to summarize."
        st.session_state.final_sentiment_display = "Neutral"
        return
    text_generator = genai.GenerativeModel('models/gemini-flash-latest')
    prompt = f"""
You are a world-class sales analyst AI. Your task is to generate a highly detailed and structured post-call analysis report.
**You must adhere strictly to the format and structure defined in the template below. Do not deviate.**

**Use the following data:**
- **CUSTOMER PROFILE:** {json.dumps(customer_profile, indent=2)}
- **FULL CALL TRANSCRIPT:** "{full_transcript}"

---

**BEGIN TEMPLATE**

# Post-Call Sales Analysis: [Populate with Customer Name] ([Populate with Customer ID])

| Metric | Detail |
| :--- | :--- |
| **Customer ID/Name** | [Populate with Customer ID / Name (Industry)] |
| **Sales Stage** | [Analyze the transcript and profile to determine the sales stage] |
| **Call Duration** | [Analyze the transcript to determine the call's length (e.g., Brief, Standard, Extended)] |
| **Product Focus** | [Analyze the transcript to identify the main product or service discussed] |
| **Analyst Recommendation** | [Based on the entire call, provide a concise, actionable recommendation] |

---

## Call Overview
[Generate a brief, 5-7 sentence high-level summary of the call's purpose, nature, and outcome here.]

---

## 1. Overall Sentiment
[Provide a rating (e.g., Positive, Neutral, Neutral/Positive) followed by a paragraph explaining the reasoning based on the customer's tone and language.]

---

## 2. Key Topics Discussed
| Topic | Detail | Analyst Note |
| :--- | :--- | :--- |
| [Identify the first key topic from the transcript] | [Detail what the customer said about this topic] | [Provide a strategic insight or note for the salesperson about this topic] |
| [Identify the second key topic from the transcript] | [Detail what the customer said about this topic] | [Provide a strategic insight or note for the salesperson about this topic] |
| [Identify the third key topic from the transcript] | [Detail what the customer said about this topic] | [Provide a strategic insight or note for the salesperson about this topic] |
*(Note: Add more rows if other substantive topics were covered, otherwise remove this note.)*

---

## 3. Objections Raised
[State "None." if no objections were raised. Follow with an explanation on why, based on the customer's clear requirements. If objections were raised, list them here.]

---

## 4. Upsell and Cross-sell Opportunities
[Write a brief introductory sentence about the opportunities presented in the call.]

| Opportunity | Strategy | Rationale |
| :--- | :--- | :--- |
| **Immediate Cross-sell:** [Identify the most immediate opportunity] | [Describe the specific strategy to capitalize on it now] | [Explain the business rationale for this strategy] |
| **Strategic Upsell:** [Identify a logical long-term upsell] | [Describe a future-focused strategy] | [Explain how this capitalizes on the customer's stated needs] |
| **Future Upsell:** [Identify a potential future opportunity] | [Describe a strategy for a subsequent interaction] | [Explain how this addresses a known customer pain point or history] |

---

## 5. Recommended Next Step
[Write a brief introductory sentence that summarizes the immediate priority.]

**Action 1 (Immediate):** [Describe the most urgent next step the salesperson must take, such as preparing a specific quote or bundle.]

**Action 2 (Follow-Up Strategy):** [Describe the action to be taken in the follow-up communication, such as introducing a service plan.]

**Action 3 (CRM Update):** [Describe the specific note or flag that should be added to the customer's file for future interactions.]

**END TEMPLATE**
"""
    try:
        response = text_generator.generate_content(prompt)
        summary = response.text.strip()
    except Exception as e:
        summary = f"Gemini API Error during summary: {e}"

    blob = TextBlob(full_transcript)
    sentiment = "Positive" if blob.sentiment.polarity > 0.1 else "Negative" if blob.sentiment.polarity < -0.1 else "Neutral"
    st.session_state.final_summary = summary
    st.session_state.final_sentiment_display = sentiment
    if not sheet_to_log: return
    try:
        header = ["Timestamp", "Customer Name", "Transcript", "Sentiment", "LLM Summary"]
        try:
            if sheet_to_log.row_values(1) != header: sheet_to_log.insert_row(header, 1)
        except gspread.exceptions.APIError: # Sheet is empty
             sheet_to_log.insert_row(header, 1)
        row_data = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), customer_profile.get('Name', 'N/A'), full_transcript, sentiment, summary]
        sheet_to_log.append_row(row_data)
        st.toast("✅ Full call summary logged to Google Sheet!")
        get_total_calls.clear()
        fetch_latest_history_from_sheet.clear()
    except Exception as e:
        st.error(f"Error logging to Google Sheet: {e}")

# --- SESSION STATE & WORKER (No changes here) ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
    st.session_state.audio_queue = queue.Queue()
    st.session_state.result_queue = queue.Queue()
    st.session_state.stop_event = threading.Event()
    st.session_state.live_chunks_display = "Load a customer profile to begin."
    st.session_state.full_transcript_parts = []
    st.session_state.final_summary = "Summary will appear here after the call."
    st.session_state.final_sentiment_display = "Neutral"
    st.session_state.customer_profile = None
    st.session_state.call_history_log = fetch_latest_history_from_sheet(sales_log_sheet)
    st.session_state.recommended_products = set()

def audio_processing_worker(audio_q, result_q, stop_ev, customer_profile, conversation_history):
    # <<< CHANGE: Initialize buffer for overlapping audio and calculate overlap in frames
    overlap_buffer = None
    OVERLAP_FRAMES = int(OVERLAP_DURATION * SAMPLE_RATE)
    
    speech_buffer, block_counter = [], 0
    while not stop_ev.is_set():
        try:
            audio_data = audio_q.get(timeout=0.1)
            speech_buffer.append(audio_data)
            block_counter += 1
            
            if block_counter >= MAX_BLOCKS_PER_CHUNK:
                current_chunk_audio = np.concatenate(speech_buffer)

                # <<< CHANGE: Prepend the previous overlap buffer to the current chunk
                if overlap_buffer is not None:
                    audio_to_process = np.concatenate([overlap_buffer, current_chunk_audio])
                else:
                    audio_to_process = current_chunk_audio

                # <<< CHANGE: Update the overlap buffer for the *next* iteration
                overlap_buffer = current_chunk_audio[-OVERLAP_FRAMES:]
                
                # Reset buffer for the next 10-second collection
                speech_buffer, block_counter = [], 0
                
                # <<< CHANGE: Process audio entirely in-memory, avoiding temp files
                try:
                    # Create an in-memory binary stream
                    mem_file = io.BytesIO()
                    # Write audio data to the in-memory file as a WAV
                    sf.write(mem_file, audio_to_process, SAMPLE_RATE, format='WAV')
                    # Rewind the "file" to the beginning so the API can read it
                    mem_file.seek(0)
                    # Some APIs require a name attribute, so we add one.
                    mem_file.name = 'live_audio.wav'

                    transcript = groq_client.audio.transcriptions.create(
                        file=mem_file,  # Pass the in-memory file object
                        model="whisper-large-v3",
                        response_format="text"
                    )

                    if transcript and transcript.strip():
                        suggestions = get_live_call_suggestions(customer_profile, conversation_history, transcript)
                        result_q.put({"type": "chunk", "transcript": transcript, "suggestions": suggestions, "timestamp": datetime.now().strftime("%H:%M:%S")})
                
                except Exception as e:
                    result_q.put({"type": "error", "message": f"Processing Error: {e}"})
                # <<< CHANGE: The 'finally' block for deleting the file is no longer needed

        except queue.Empty:
            continue

# --- UI (No changes here) ---
with st.sidebar:
    st.title("AI Sales Co-Pilot")
    st.info(f"LLM in use: **{st.session_state.get('active_llm', 'N/A')}**")
    st.subheader("Prepare for Your Call")
    customer_email = st.text_input("Enter Customer Email", help="e.g., alex.r@techinn.com")
    if st.button("Load Customer Profile"):
        with st.spinner("Fetching full customer profile from CRM..."):
            profile, error = fetch_full_customer_profile(crm_customers_sheet, crm_products_sheet, crm_interactions_sheet, customer_email)
            if error: st.error(error); st.session_state.customer_profile = None
            else: st.session_state.customer_profile = profile; st.success(f"Loaded profile for {profile.get('Name')}")
    if st.session_state.customer_profile:
        with st.expander("Customer Briefing", expanded=True):
            profile = st.session_state.customer_profile
            st.markdown(f"**Name:** {profile.get('Name', 'N/A')}")
            st.markdown(f"**Interests:** {profile.get('Preferences_Interests', 'N/A')}")
            st.markdown(f"**Purchase History:** {profile.get('purchase_history_summary', 'N/A')}")
            st.markdown("**Interaction History:**")
            for interaction in profile.get('interaction_history', []):
                st.caption(f"- {interaction['date']}: {interaction['outcome']} ({interaction['notes']})")

welcome_message = "Hello, Welcome!"
if st.session_state.customer_profile:
    welcome_message = f"Hello, Welcome {st.session_state.customer_profile.get('Name', '')}!"
st.title(welcome_message)

cols = st.columns([1, 1, 5])
with cols[0]:
    start_disabled = st.session_state.is_running or not st.session_state.customer_profile
    if st.button("▶️ Start Call", disabled=start_disabled, use_container_width=True):
        st.session_state.is_running = True; st.session_state.stop_event.clear()
        st.session_state.live_chunks_display = "🟢 Listening... Speak into your microphone."
        st.session_state.full_transcript_parts = []; st.session_state.recommended_products = set()
        q = st.session_state.audio_queue
        def audio_callback(indata, frames, time, status): q.put(indata.copy())
        st.session_state.stream = sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCKSIZE)
        st.session_state.stream.start()
        st.session_state.worker_thread = threading.Thread(
            target=audio_processing_worker,
            args=(q, st.session_state.result_queue, st.session_state.stop_event, 
                  st.session_state.customer_profile, st.session_state.full_transcript_parts))
        st.session_state.worker_thread.start()
        st.rerun()
with cols[1]:
    if st.button("⏹️ Stop Call", disabled=not st.session_state.is_running, use_container_width=True):
        st.session_state.is_running = False; st.session_state.stop_event.set()
        if 'worker_thread' in st.session_state and st.session_state.worker_thread.is_alive(): st.session_state.worker_thread.join(timeout=2)
        if 'stream' in st.session_state: st.session_state.stream.stop(); st.session_state.stream.close()
        with st.spinner("Generating final, personalized call summary..."):
            full_transcript = " ".join(st.session_state.full_transcript_parts)
            perform_final_analysis_and_log(sales_log_sheet, st.session_state.customer_profile, full_transcript)
        st.rerun()

st.header("Live Call Dashboard")
st.subheader(f"Live Co-Pilot Feed {'(Listening...)' if st.session_state.is_running else '(Not Active)'}")
st.text_area("Live Feed", value=st.session_state.live_chunks_display, height=300, key="live_feed")
st.header("Post-Call Analysis")
st.subheader("Products Recommended During Call")
if 'recommended_products' in st.session_state and st.session_state.recommended_products:
    for product in st.session_state.recommended_products:
        st.markdown(f"- {product}")
else:
    st.caption("No specific products were recommended by the AI during this call.")
st.subheader("Final Call Summary")
sentiment_color = {"Positive": "inverse", "Negative": "off"}.get(st.session_state.final_sentiment_display, "normal")
st.metric("Overall Sentiment", st.session_state.final_sentiment_display, delta_color=sentiment_color)
st.info(f"**Overall Summary:**\n\n{st.session_state.final_summary}")

if st.session_state.is_running:
    try:
        result = st.session_state.result_queue.get(block=False)
        if result["type"] == "chunk":
            suggestions = result.get("suggestions", {})
            if "error" in suggestions: st.error(f"AI Suggestion Error: {suggestions['error']}")
            else:
                st.session_state.full_transcript_parts.append(result["transcript"])
                recommendation = suggestions.get("product_recommendation", "")
                if recommendation: st.session_state.recommended_products.add(recommendation)
                sentiment = suggestions.get("sentiment", "Neutral")
                step = suggestions.get("next_step_suggestion", "")
                emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐", "Questioning": "🤔"}.get(sentiment, "💬")
                new_chunk = f"[{result['timestamp']}] {emoji} Customer: {result['transcript']}"
                if recommendation: new_chunk += f"\n\n🤖 Recommendation: {recommendation}"
                if step: new_chunk += f"\n\n🤖 Next Step: {step}"
                if st.session_state.live_chunks_display.startswith("🟢 Listening..."): st.session_state.live_chunks_display = new_chunk
                else: st.session_state.live_chunks_display = new_chunk + "\n\n---\n\n" + st.session_state.live_chunks_display
        elif result["type"] == "error": st.error(result["message"])
        st.rerun()
    except queue.Empty: time.sleep(0.5); st.rerun()

st.divider()
total_calls = get_total_calls(sales_log_sheet)
st.subheader(f"Recent Call History ({total_calls} Total Calls Logged)")
if 'call_history_log' not in st.session_state: st.session_state.call_history_log = fetch_latest_history_from_sheet(sales_log_sheet)
if not st.session_state.call_history_log:
    st.caption("History of past calls will appear here after you log a call.")
else:
    for call in st.session_state.call_history_log:
        with st.expander(f"**{call['timestamp']}** | Overall Sentiment: **{call['sentiment']}**"):
            st.markdown(f"**LLM Summary:** {call['summary']}")
            st.caption("**Full Transcript:**"); st.write(f"_{call['transcript']}_")
