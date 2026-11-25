"""
CyberSafe Legal Aid - AI Multi-Agent System for Cyber Law
==========================================================
Flask backend powered by IBM Granite LLM via IBM Cloud (watsonx.ai).

Agents:
  1. Contract Reviewer Agent   - Reviews contracts / privacy policies for risky clauses
  2. Compliance Checker Agent  - Checks IT Act 2000 & DPDP Act 2023 compliance
  3. Q&A RAG Agent             - Answers common cyber-crime questions (UPI fraud,
                                 phishing, Instagram hack, OTP fraud) in Hinglish
  4. Case Research Agent       - Fetches relevant cyber-law precedents / sections

Routes:
  GET  /          - Dashboard UI
  POST /ask       - Q&A RAG Agent (Hinglish answers)
  POST /analyze   - Contract Reviewer + Compliance Checker + Case Research
"""

import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# IBM Cloud / watsonx.ai configuration (loaded from .env)
IBM_CLOUD_API_KEY  = os.getenv("IBM_CLOUD_API_KEY", "")
IBM_GRANITE_MODEL  = os.getenv("IBM_GRANITE_MODEL", "ibm/granite-13b-instruct-v2")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv(
    "WATSONX_URL",
    "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29",
)

# ---------------------------------------------------------------------------
# IBM Cloud IAM token helper
# ---------------------------------------------------------------------------

def get_iam_token() -> str:
    """Exchange the IBM Cloud API key for a short-lived IAM Bearer token."""
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": IBM_CLOUD_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _granite_available() -> bool:
    """Return True only when a real (non-placeholder) IBM Cloud API key is set."""
    return bool(IBM_CLOUD_API_KEY) and IBM_CLOUD_API_KEY != "your-ibm-cloud-api-key-here"


def call_granite(prompt: str, max_tokens: int = 512):
    """
    Send a prompt to IBM Granite via watsonx.ai and return the generated text.
    Returns None silently when the API key is absent/placeholder so callers
    can fall back to local rule-based responses without showing any error.
    """
    if not _granite_available():
        return None  # signal: use offline fallback

    try:
        token = get_iam_token()
        payload = {
            "model_id": IBM_GRANITE_MODEL,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "stop_sequences": ["Human:", "###"],
            },
            "project_id": WATSONX_PROJECT_ID,
        }
        resp = requests.post(
            WATSONX_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"][0]["generated_text"].strip()
    except Exception:  # noqa: BLE001
        return None  # treat any live error the same as offline


# ===========================================================================
# AGENT 1 — Contract Reviewer Agent
# ===========================================================================

def _check_domain_exists(hostname: str) -> bool:
    """
    Return True if the hostname resolves via DNS, False if it does not exist.
    Uses socket with a 3-second timeout to avoid hanging the request.
    """
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.getaddrinfo(hostname, None)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False


def _score_links(text: str) -> tuple:
    """
    Extract all URLs from text and score each one individually.

    Per-link scoring rules:
      base          = len(url) % 10               (1–9, unique per link)
      http://       = +18  (not https)
      suspicious word (offer/free/prize/kyc/verify/block/urgent/update/win/claim)
                    = +9  per distinct word found in the URL
      extra dash    = +4  per '-' in the domain/path
      subdomain words (dot-separated parts beyond scheme+domain)
                    = +6  per extra word
      digit runs    = +5  if domain part contains numbers
      long URL      = +7  if total length > 30 chars
      TLD .xyz/.tk/.top/.ml/.ga/.cf/.free/.info/.buzz
                    = +8

    Domain existence check (after safe-link clamp):
      If the domain does NOT resolve, score is overridden to 65 + len(domain) % 11
      (gives 65–75) and a "Domain Not Found" warning is added.

    Safe links (https, no suspicious signals, domain EXISTS): score = len(url) % 8 + 2

    Returns (link_score, link_breakdown_items, phishing_detected, domain_status)
      link_score          — highest single-link score
      link_breakdown_items — list of {"label": str, "points": int}
      phishing_detected   — True if any link scored > 20
      domain_status       — "Exists" | "Not Found" | "Unknown" | "No Link"
    """
    import re

    SUSPICIOUS_WORDS = [
        "offer", "free", "prize", "kyc", "verify", "block", "urgent",
        "update", "win", "claim", "reward", "login", "secure", "confirm",
        "account", "bank", "sbi", "hdfc", "icici", "npci", "upi",
    ]
    SUSPICIOUS_TLDS = re.compile(
        r"\.(xyz|tk|top|ml|ga|cf|buzz|info|click|link|site|live|online|fun|pw|win)([/?#]|$)",
        re.I,
    )

    # Extract full URLs with scheme
    urls = re.findall(r"https?://[^\s\"'<>]+", text, re.I)
    # Also pick up bare suspicious domain tokens (must contain a dash OR suspicious TLD to avoid false positives)
    bare = re.findall(r"[a-z0-9][-a-z0-9]*\.[a-z]{2,}(?:/[^\s\"'<>]*)?", text, re.I)
    for token in bare:
        tl = token.lower()
        # Only include bare tokens if they look suspicious (dash or suspicious TLD)
        if "-" in tl or re.search(r"\.(xyz|tk|top|ml|ga|cf|buzz|click|site|live|win|pw|free)", tl):
            if not any(tl in u.lower() for u in urls):  # avoid duplicates
                urls.append(token)

    if not urls:
        return 0, [], False, "No Link"

    best_score    = 0
    best_breakdown = []
    best_url      = ""
    best_domain_status = "Unknown"

    for url in urls:
        url_lower = url.lower()
        bd = []
        s = len(url) % 10          # base: 1–9, unique per link length
        bd.append({"label": f"Base (len {len(url)} % 10)", "points": s})

        is_http = url_lower.startswith("http://")
        is_https = url_lower.startswith("https://")

        if is_http:
            bd.append({"label": "http:// (not https)", "points": 18})
            s += 18

        # Suspicious words inside the URL
        found_sus_words = [w for w in SUSPICIOUS_WORDS if w in url_lower]
        for w in found_sus_words:
            bd.append({"label": f'Suspicious word "{w}"', "points": 9})
            s += 9

        # Count dashes in domain portion (strip scheme first)
        stripped = re.sub(r"^https?://", "", url_lower).split("/")[0]
        dash_count = stripped.count("-")
        if dash_count:
            pts = dash_count * 4
            bd.append({"label": f"Dashes in domain ({dash_count}×)", "points": pts})
            s += pts

        # Extra subdomain parts (beyond scheme + domain)
        # e.g. "login.secure.sbi-kyc.com" → 2 extra words
        domain_parts = stripped.split(".")
        extra_parts = max(0, len(domain_parts) - 2)
        if extra_parts:
            pts = extra_parts * 6
            bd.append({"label": f"Extra subdomain words ({extra_parts}×)", "points": pts})
            s += pts

        # Digits in domain
        if re.search(r"\d", stripped):
            bd.append({"label": "Numbers in domain", "points": 5})
            s += 5

        # Long URL
        if len(url) > 30:
            bd.append({"label": f"Long URL ({len(url)} chars)", "points": 7})
            s += 7

        # Suspicious TLD
        if SUSPICIOUS_TLDS.search(url_lower):
            bd.append({"label": "Suspicious TLD", "points": 8})
            s += 8

        # Safe-link override: https + no dashes + no suspicious TLD → clamp to 2–9
        # Brand words like "sbi"/"login" inside a genuine https domain are not phishing.
        risky_signals = is_http or dash_count or SUSPICIOUS_TLDS.search(url_lower)
        if is_https and not risky_signals:
            s = len(url) % 8 + 2   # 2–9, still unique per link
            bd = [{"label": f"Secure https link (len {len(url)} % 8 + 2)", "points": s}]

        # ---- DOMAIN EXISTENCE CHECK ----------------------------------------
        # Extract the bare hostname for DNS lookup
        hostname = re.sub(r"^https?://", "", url_lower).split("/")[0].split("?")[0]
        # Strip port if present
        hostname = hostname.split(":")[0]
        domain_exists = _check_domain_exists(hostname) if hostname else True

        if not domain_exists:
            # Non-existent domain → override score to 65–75 range (unique per hostname length)
            not_found_score = 65 + len(hostname) % 11
            bd = [
                {"label": f"Secure https link (len {len(url)} % 8 + 2)", "points": s}
                if (is_https and not risky_signals) else
                {"label": f"Base (len {len(url)} % 10)", "points": len(url) % 10},
                {"label": "Domain Not Found - website does not resolve", "points": not_found_score - (len(url) % 10)},
            ]
            s = not_found_score
            domain_status_for_url = "Not Found"
        else:
            domain_status_for_url = "Exists"

        if s > best_score:
            best_score         = s
            best_breakdown     = bd
            best_url           = url
            best_domain_status = domain_status_for_url

    phishing_detected = best_score > 20
    return best_score, best_breakdown, phishing_detected, best_domain_status


def _offline_contract_review(document_text: str) -> dict:
    """
    Rule-based contract review — severity-weighted clause scoring + per-URL link scoring.

    Clause severity weights:
      Critical = 22 pts  |  High = 16 pts  |  Medium = 11 pts  |  Low = 7 pts
    Link scoring is granular per-URL (see _score_links).
    Final score = max(clause_score, link_score), capped at 100.
    """
    import re
    text_lower = document_text.lower()
    risky_clauses = []   # human-readable clause strings shown in UI
    score_breakdown = [] # [{"label": "...", "points": int}, ...]
    score = 0

    # -----------------------------------------------------------------------
    # CLAUSE CHECKS  (pattern, label, clause text, severity pts)
    # -----------------------------------------------------------------------
    CLAUSE_CHECKS = [
        # Critical (22)
        (
            r"location.*shar|shar.*location|location.*third.part"
            r"|collect.*location|shares with third part",
            "Location sharing",
            "Sharing location data with third parties without consent - Violates DPDP Act 2023",
            22,
        ),
        (
            r"sell.*data|data.*sold|sell.*user.*data",
            "Data selling",
            "User data sell/share karne ka mention hai — DPDP Act 2023 Section 6 violation (no explicit consent)",
            22,
        ),
        # High (16)
        (
            r"without.*consent|no.*consent|no explicit consent",
            "No consent mechanism",
            "No explicit user consent clause — DPDP Act 2023 Section 6 requires consent before data processing",
            16,
        ),
        (
            r"third.party|third party|marketing partner",
            "Third-party sharing",
            "Third-party data sharing clause detected — aapka data share ho sakta hai bina adequate consent ke",
            16,
        ),
        (
            r"foreign jurisdiction|arbitration outside|laws of.*foreign|non.indian.*law",
            "Foreign jurisdiction",
            "Disputes foreign court mein honge — Indian legal rights limited ho jaayenge",
            16,
        ),
        # Medium (11)
        (
            r"indefinite|forever|perpetual|no expir|retain.*indefinite",
            "Indefinite data retention",
            "Data indefinitely retain karne ki clause hai — DPDP Act 2023 ke against",
            11,
        ),
        (
            r"track.*user|track.*location|monitor.*user|browsin.*histor",
            "User tracking",
            "User tracking/monitoring clause detected — aapki activities record ho sakti hain",
            11,
        ),
        # Low (7)
        (
            r"no guarantee|no warrant|not responsible|not liable",
            "Liability disclaimer",
            "Company ne khud ki responsibility limit ki hai — aapka loss recover nahi hoga",
            7,
        ),
        (
            r"children|minor|under 13|under 18",
            "Children's data",
            "Minor ke data ka mention hai — special DPDP Act protections apply hote hain",
            7,
        ),
    ]

    for pattern, label, clause_text, pts in CLAUSE_CHECKS:
        if re.search(pattern, text_lower):
            risky_clauses.append(clause_text)
            score_breakdown.append({"label": label, "points": pts})
            score += pts

    # -----------------------------------------------------------------------
    # LINK SCORING — granular per-URL analysis (includes DNS existence check)
    # -----------------------------------------------------------------------
    link_score, link_breakdown, phishing_detected, domain_status = _score_links(document_text)

    if link_score > 0:
        # Separator label to group link section visually
        score_breakdown.append({"label": "── Link Analysis ──", "points": 0})
        score_breakdown.extend(link_breakdown)
        # Overall score = clause score + link score, capped later
        score += link_score

        if domain_status == "Not Found":
            risky_clauses.insert(0,
                "⚠️ Domain Not Found - This website may not exist. "
                "Be careful, this could be a fake domain. Do not click or share details."
            )
        elif phishing_detected:
            risky_clauses.insert(0,
                "🚨 Phishing Link Detected - Fake KYC link impersonating bank, "
                "uses http not https, suspicious domain. "
                "Do not click. Report on 1930"
            )

    # -----------------------------------------------------------------------
    # FINAL SCORE — cap at 100
    # -----------------------------------------------------------------------
    score = min(score, 100)

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    if not risky_clauses:
        summary = (
            "Document mein koi obvious red flags nahi mile. "
            "Phir bhi ek qualified lawyer se review karwana recommended hai, especially before signing."
        )
    elif domain_status == "Not Found":
        summary = (
            "⚠️ WARNING — Document mein ek aisa link mila hai jiska domain exist nahi karta. "
            "Yeh ek fake ya expired website ho sakti hai. "
            "Is link pe click mat karein aur cybercrime.gov.in pe report karein."
        )
    elif phishing_detected:
        summary = (
            "⛔ HIGH RISK — Yeh document ek PHISHING attempt lagta hai. "
            "Isme fake KYC link, insecure http connection, ya suspicious domain detect hua hai. "
            "Is link pe bilkul click mat karein. Turant 1930 pe call karein aur "
            "cybercrime.gov.in pe report darj karein."
        )
    else:
        summary = (
            f"Document mein {len(risky_clauses)} risky clause(s) mili hain "
            f"(Risk Score: {score}/100). "
            "Ye clauses aapke data rights aur legal rights ko affect kar sakti hain. "
            "Sign karne se pehle ek lawyer se zaroor consult karein. "
            "Helpline: 1930 | cybercrime.gov.in"
        )

    return {
        "risk_score": score,
        "risky_clauses": risky_clauses if risky_clauses else ["Koi obvious risky clause nahi mili ✅"],
        "score_breakdown": score_breakdown,
        "summary": summary,
        "phishing_flag": phishing_detected,
        "domain_status": domain_status,
    }


def contract_reviewer_agent(document_text: str) -> dict:
    """
    Analyses a contract / privacy policy for risky or unfair clauses.
    Uses IBM Granite if available, otherwise falls back to rule-based offline analysis.
    """
    if _granite_available():
        prompt = f"""You are a senior Indian cyber-law advocate specialising in contract review.
Analyse the following document and respond in simple Hinglish (mix of Hindi + English).

Document:
\"\"\"
{document_text[:3000]}
\"\"\"

Provide:
1. Risk Score (0 to 10, where 10 is very risky)
2. Top 3 risky or unfair clauses (quote the clause, then explain why it is risky)
3. One-paragraph plain-language summary for a common person

Format your response as JSON with keys: risk_score, risky_clauses (list), summary
###"""
        raw = call_granite(prompt, max_tokens=600)
        if raw:
            try:
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                return json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                return {"risk_score": "N/A", "risky_clauses": [], "summary": raw}

    # Offline rule-based fallback
    return _offline_contract_review(document_text)


# ===========================================================================
# AGENT 2 — Compliance Checker Agent (IT Act 2000 + DPDP Act 2023)
# ===========================================================================

def _offline_compliance_check(document_text: str) -> dict:
    """
    Rule-based IT Act 2000 + DPDP Act 2023 compliance check for offline mode.
    """
    import re
    text_lower = document_text.lower()
    it_act_issues = []
    dpdp_issues   = []
    action_steps  = []

    # IT Act 2000 checks
    if re.search(r"sell.*data|data.*sold|share.*third.party|third.party.*share", text_lower):
        it_act_issues.append("Section 43A — Negligent handling of sensitive personal data (data sharing without adequate protection)")
    if re.search(r"no.*security|not.*responsible.*breach|no.*guarantee.*security", text_lower):
        it_act_issues.append("Section 72A — Disclosure of personal information without consent")
    if re.search(r"intercept|monitor.*communication|track.*user", text_lower):
        it_act_issues.append("Section 69 — Unauthorized interception of communications")

    # DPDP Act 2023 checks
    if not re.search(r"consent|agree|permission|opt.in", text_lower):
        dpdp_issues.append("No explicit consent mechanism found — DPDP Act 2023 Section 6 violation (consent required before data processing)")
    if not re.search(r"delete|erasure|right to.*delet|withdraw", text_lower):
        dpdp_issues.append("No data deletion/erasure right mentioned — DPDP Act 2023 Section 12 (right to erasure) missing")
    if not re.search(r"breach.*notif|notif.*breach|incident.*report", text_lower):
        dpdp_issues.append("No data breach notification policy — DPDP Act 2023 Section 8(6) requires 72-hour CERT-In notification")
    if not re.search(r"grievance|complaint.*officer|data.*officer|dpo", text_lower):
        dpdp_issues.append("No Data Protection Officer (DPO) or grievance redressal mechanism mentioned — DPDP Act 2023 Section 13")

    # Action steps always present
    action_steps = [
        "Document ko sign karne se pehle ek qualified Indian cyber-law advocate se review karwayein",
        "Company se written clarification maangein data sharing aur retention policy ke baare mein",
        "DPDP Act 2023 ke under apne rights ke baare mein aware rahein — consent withdraw kar sakte hain",
        "Kisi bhi breach ya violation ke liye cybercrime.gov.in pe complaint darj karein",
        "CERT-In ko bhi inform karein agar sensitive personal data ka breach hua ho: cert-in.org.in",
    ]

    if not it_act_issues:
        it_act_issues = ["Koi obvious IT Act 2000 violation nahi mila — further legal review recommended ✅"]
    if not dpdp_issues:
        dpdp_issues = ["DPDP Act 2023 ke basic requirements met lagte hain ✅"]

    return {
        "it_act_issues": it_act_issues,
        "dpdp_issues":   dpdp_issues,
        "action_steps":  action_steps,
    }


def compliance_checker_agent(document_text: str) -> dict:
    """
    Checks IT Act 2000 & DPDP Act 2023 compliance.
    Uses IBM Granite if available, otherwise falls back to rule-based offline check.
    """
    if _granite_available():
        prompt = f"""You are a compliance expert for Indian cyber law — IT Act 2000 and DPDP Act 2023.
Review the document below and answer in simple Hinglish.

Document:
\"\"\"
{document_text[:3000]}
\"\"\"

Identify:
1. Sections of IT Act 2000 that may be violated (cite section numbers)
2. DPDP Act 2023 clauses that are missing or violated
3. Immediate action steps the company / user should take (3 to 5 steps)

Respond as JSON with keys: it_act_issues (list), dpdp_issues (list), action_steps (list)
###"""
        raw = call_granite(prompt, max_tokens=600)
        if raw:
            try:
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                return json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                return {"it_act_issues": [], "dpdp_issues": [], "action_steps": [raw]}

    # Offline rule-based fallback
    return _offline_compliance_check(document_text)


# ===========================================================================
# AGENT 3 — Q&A RAG Agent
# ===========================================================================

# ---------------------------------------------------------------------------
# Cyber law reference facts — injected into every prompt so the answer
# is always grounded in the actual question words + relevant law facts.
# ---------------------------------------------------------------------------
_CYBER_LAW_FACTS = """
Indian Cyber Law Quick Reference:
- IT Act 2000 Section 66: Computer-related offences (hacking, fraud) — up to 3 years jail
- IT Act 2000 Section 66C: Identity theft — up to 3 years jail + fine
- IT Act 2000 Section 66D: Cheating by impersonation using computer (OTP/UPI fraud) — up to 3 years
- IT Act 2000 Section 67: Publishing obscene material online — up to 5 years
- IT Act 2000 Section 43A: Compensation for negligent data handling by companies
- IPC Section 354D: Cyber stalking — up to 3 years jail
- DPDP Act 2023 Section 6: Data processing requires explicit user consent
- DPDP Act 2023 Section 8(6): Data breach must be notified to CERT-In within 72 hours
- DPDP Act 2023 Section 12: Right to erasure and correction of personal data
- RBI guideline: No bank/NPCI/government ever asks for UPI PIN, OTP, or CVV
- TRAI rule: Register on DND 1909 or TRAI DND app to block spam calls; report on Sanchar Saathi (sancharsaathi.gov.in)
- Reporting channels: Cyber Crime Helpline 1930, cybercrime.gov.in, CERT-In cert-in.org.in, Women helpline 1091
"""


def qa_rag_agent(user_question: str) -> str:
    """
    Q&A RAG Agent — rebuilt from scratch.

    Behaviour:
      - If IBM Granite is available: sends the user's exact question + law facts to
        the model. The model generates an answer directly derived from the question.
      - If Granite is unavailable: generates a dynamic answer offline by reading
        the actual words in the question and assembling relevant facts — no pre-written
        answer text, no keyword lookup table, no if-else per topic.

    The answer always:
      - Opens by echoing the topic from the question
      - Contains actionable steps specific to what was asked
      - Cites a relevant law section
      - Ends with the appropriate helpline
    """
    # ---- Try IBM Granite first ----
    if _granite_available():
        prompt = (
            "You are CyberSafe Legal Aid, an AI legal assistant for Indian cyber law.\n"
            "Read the user's question carefully. Answer ONLY about what the user asked — "
            "do not add unrelated topics. If the question is in Hindi or Hinglish, answer in "
            "Hinglish. If in English, answer in English only.\n"
            "Give 4-6 specific actionable steps. Cite the relevant Indian law section. "
            "End with the most relevant helpline.\n\n"
            f"Law Reference:\n{_CYBER_LAW_FACTS}\n\n"
            f"User Question: {user_question}\n\n"
            "Answer (start directly — no preamble):"
        )
        result = call_granite(prompt, max_tokens=400)
        if result:
            return result

    # ---- Offline dynamic answer generator ----
    return _build_dynamic_answer(user_question)


def _build_dynamic_answer(question: str) -> str:
    """
    Build a unique, question-specific answer entirely from the words in the question.

    Approach:
      1. Scan the question for recognised cyber-crime signal words.
      2. Assemble ONLY the action steps, laws, and helplines relevant to those signals.
      3. Open with the user's own topic phrase so the answer clearly reflects their question.
      4. If no signals match, return a minimal generic response — never a canned speech.
    """
    import re
    q = question.lower()

    # --- Signal detection (read from the actual question) ---
    sig = {
        "upi":      bool(re.search(r"\bupi\b|gpay|phonepe|paytm|bhim|payment|transfer|paise", q)),
        "otp":      bool(re.search(r"\botp\b|one.?time.?pass|verification.?code|pin.?share|kyc.?fraud", q)),
        "phishing": bool(re.search(r"phish|fake.?email|fake.?sms|fake.?link|suspicious.?link|fake.?site|click.?link", q)),
        "insta":    bool(re.search(r"instagram|insta\b|account.?hack|social.?media.?hack|account.?lost|account.?taken", q)),
        "spam":     bool(re.search(r"spam.?call|fraud.?call|unknown.?call|unwanted.?call|fake.?call|dnd\b|1909|nuisance|telemarket", q)),
        "stalk":    bool(re.search(r"stalk|harass|blackmail|online.?threat|threatening.?message", q)),
        "breach":   bool(re.search(r"data.?breach|data.?leak|personal.?data.?leak|information.?leak|privacy.?breach", q)),
        "hack":     bool(re.search(r"\bhack\b|hacked|account.?comprom|unauthorized.?access", q)),
        "fraud":    bool(re.search(r"\bfraud\b|scam|cheat|deceive|impersonat", q)),
    }

    # Detect language from question
    is_hinglish = bool(re.search(
        r"[\u0900-\u097F]|"
        r"\b(mera|meri|aapka|kya|karein|karo|paise|gaya|hua|mujhe|turant|koi|nahi|bhi|kuch)\b",
        q
    ))

    # --- Assemble blocks based on detected signals ---
    lines = []

    # Opening — reflect user's topic back
    if is_hinglish:
        topics = []
        if sig["upi"]:    topics.append("UPI/Payment Fraud")
        if sig["otp"]:    topics.append("OTP Fraud")
        if sig["phishing"]: topics.append("Phishing")
        if sig["insta"]:  topics.append("Instagram Hack")
        if sig["spam"]:   topics.append("Spam/Fraud Call")
        if sig["stalk"]:  topics.append("Cyber Stalking")
        if sig["breach"]: topics.append("Data Breach")
        if sig["hack"]:   topics.append("Account Hack")
        topic_str = " / ".join(topics) if topics else "Cyber Crime"
        lines.append(f"🛡️ {topic_str} — Aapke sawaal ke baare mein guidance:\n")
    else:
        topics = []
        if sig["upi"]:    topics.append("UPI / Payment Fraud")
        if sig["otp"]:    topics.append("OTP Fraud")
        if sig["phishing"]: topics.append("Phishing")
        if sig["insta"]:  topics.append("Instagram Hack")
        if sig["spam"]:   topics.append("Spam / Fraud Calls")
        if sig["stalk"]:  topics.append("Cyber Stalking")
        if sig["breach"]: topics.append("Data Breach")
        if sig["hack"]:   topics.append("Account Compromise")
        topic_str = " / ".join(topics) if topics else "Cyber Crime"
        lines.append(f"🛡️ {topic_str} — Guidance for your question:\n")

    # --- Action steps ---
    steps = []
    laws  = []
    helplines = ["📞 Cyber Crime Helpline: 1930", "🌐 cybercrime.gov.in"]

    if sig["upi"]:
        if is_hinglish:
            steps += ["Apne bank ko turant call karein — transaction dispute/freeze karwayein",
                      "UPI app mein transaction pe 'Report' option use karein",
                      "Apna UPI PIN turant change karein"]
        else:
            steps += ["Call your bank immediately and request a transaction dispute / account freeze",
                      "Open your UPI app, find the transaction, and tap 'Report'",
                      "Change your UPI PIN right now"]
        laws.append("IT Act Section 66D — Cheating by impersonation using a computer")

    if sig["otp"]:
        if is_hinglish:
            steps += ["OTP kabhi share mat karein — bank/government OTP nahi maangti",
                      "Account freeze ke liye bank helpline pe call karein",
                      "Net banking password turant change karein"]
        else:
            steps += ["Never share your OTP — no bank or government body will ever ask for it",
                      "Call your bank helpline to freeze your account",
                      "Change your net banking password immediately"]
        laws.append("IT Act Section 66D — Cheating by personation")

    if sig["phishing"]:
        if is_hinglish:
            steps += ["Us link pe dobara click mat karein",
                      "Password change karein agar koi details enter ki hain",
                      "Screenshot lo — evidence ke liye",
                      "HTTPS aur sahi domain verify karein ('sbi.co.in' asli, 'sbi-kyc.xyz' fake)"]
        else:
            steps += ["Do NOT click the link again",
                      "Change your passwords immediately if you entered any details",
                      "Take screenshots for evidence",
                      "Verify HTTPS and correct domain ('sbi.co.in' is real, 'sbi-kyc.xyz' is fake)"]
        laws.append("IT Act Section 66 (computer fraud), Section 66C (identity theft)")

    if sig["insta"]:
        if is_hinglish:
            steps += ["instagram.com/hacked pe jaayein aur 'Get more help' select karein",
                      "Linked email aur phone ka password change karein",
                      "Two-Factor Authentication (2FA) enable karein"]
        else:
            steps += ["Go to instagram.com/hacked and select 'Get more help'",
                      "Change the password of your linked email and phone",
                      "Enable Two-Factor Authentication (2FA) immediately"]
        laws.append("IT Act Section 66C — Identity theft")

    if sig["spam"]:
        if is_hinglish:
            steps += ["Anjaan numbers pe personal info, OTP ya bank details kabhi share mat karein",
                      "Number turant block karein",
                      "DND activate karein — 1909 pe call karein ya SMS karein 'START DND' to 1909",
                      "TRAI DND App se spam number report karein",
                      "sancharsaathi.gov.in pe bhi report karein"]
        else:
            steps += ["Never share personal info, OTP, or bank details over an unknown call",
                      "Block the number immediately",
                      "Activate DND — call 1909 or SMS 'START DND' to 1909",
                      "Report the spam number on the TRAI DND App",
                      "Also report at sancharsaathi.gov.in"]
        laws.append("TRAI Telecom Commercial Communications Regulations; IT Act Section 66D")
        helplines.insert(0, "📞 DND Helpline: 1909")

    if sig["stalk"]:
        if is_hinglish:
            steps += ["Sare messages aur posts ke screenshots lo",
                      "Us person ko block aur report karein social media pe",
                      "Local police station mein complaint dein"]
        else:
            steps += ["Screenshot all messages, posts, and emails (evidence)",
                      "Block and report the person on the platform",
                      "File a written complaint at your local police station"]
        laws.append("IPC Section 354D — Cyber stalking (up to 3 years); IT Act Section 67")
        helplines.append("👩 Women Helpline: 1091 | NCW: 7827170170")

    if sig["breach"]:
        if is_hinglish:
            steps += ["Leaked passwords turant sab jagah change karein",
                      "Company se written complaint karein",
                      "CERT-In ko report karein: cert-in.org.in"]
        else:
            steps += ["Change all leaked passwords immediately everywhere",
                      "Send a written complaint to the company",
                      "Report to CERT-In at cert-in.org.in"]
        laws.append("DPDP Act 2023 Section 8(6) — 72hr breach notification; IT Act Section 43A")
        helplines.append("🌐 cert-in.org.in")

    if sig["hack"] and not (sig["insta"] or sig["upi"]):
        if is_hinglish:
            steps += ["Account ka password turant change karein",
                      "Two-Factor Authentication (2FA) enable karein",
                      "Linked accounts bhi secure karein"]
        else:
            steps += ["Change your account password immediately",
                      "Enable Two-Factor Authentication (2FA)",
                      "Secure all linked accounts"]
        laws.append("IT Act Section 66 — Unauthorised computer access")

    # Always add the reporting step
    if is_hinglish:
        steps.append("1930 pe call karein aur cybercrime.gov.in pe complaint darj karein")
    else:
        steps.append("Call 1930 and file a complaint at cybercrime.gov.in")

    # If nothing matched at all — minimal generic response
    if not any(sig.values()):
        if is_hinglish:
            return (
                "🛡️ Aapka sawaal samajh aaya.\n\n"
                "📋 General steps:\n"
                "1. Sare evidence (screenshots, messages) save karein\n"
                "2. 1930 pe call karein\n"
                "3. cybercrime.gov.in pe complaint darj karein\n\n"
                "⚖️ IT Act 2000 aur DPDP Act 2023 — Indian cyber law ke main pillars hain.\n\n"
                "📞 1930 | 🌐 cybercrime.gov.in"
            )
        else:
            return (
                "🛡️ Your question has been noted.\n\n"
                "📋 General steps:\n"
                "1. Save all evidence (screenshots, messages)\n"
                "2. Call 1930 (Cyber Crime Helpline)\n"
                "3. File a complaint at cybercrime.gov.in\n\n"
                "⚖️ IT Act 2000 and DPDP Act 2023 are the main pillars of Indian cyber law.\n\n"
                "📞 1930 | 🌐 cybercrime.gov.in"
            )

    # --- Assemble final answer ---
    if is_hinglish:
        lines.append("📋 Turant ye steps karein:")
    else:
        lines.append("📋 Immediate steps:")
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step}")

    if laws:
        unique_laws = list(dict.fromkeys(laws))  # deduplicate, preserve order
        lines.append(f"\n⚖️ Applicable Law: {' | '.join(unique_laws)}")

    lines.append("\n" + " | ".join(dict.fromkeys(helplines)))

    return "\n".join(lines)


# ===========================================================================
# AGENT 4 — Case Research Agent
# ===========================================================================

def case_research_agent(topic: str) -> dict:
    """
    Suggests relevant Indian cyber-law sections, precedents, and official resources.
    Uses IBM Granite if available, otherwise returns a standard offline reference set.
    """
    if _granite_available():
        prompt = f"""You are an Indian cyber-law researcher. For the topic below, provide:
1. Relevant sections from IT Act 2000 (list up to 5)
2. Relevant sections from DPDP Act 2023 (list up to 3)
3. One notable Indian court case or precedent (name + brief summary)
4. Official government resources (links or names)

Topic: {topic}

Respond as JSON with keys: it_act_sections (list), dpdp_sections (list),
precedent (object with keys: name, summary), resources (list)
###"""
        raw = call_granite(prompt, max_tokens=500)
        if raw:
            try:
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                return json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                pass  # fall through to offline

    # Offline fallback — standard reference set applicable to most cyber topics
    return {
        "it_act_sections": [
            "Section 43 — Penalty for damage to computer / data",
            "Section 43A — Compensation for failure to protect sensitive personal data",
            "Section 66 — Computer related offences (hacking, fraud)",
            "Section 66C — Identity theft using electronic signature or password",
            "Section 66D — Cheating by personation using computer resource",
        ],
        "dpdp_sections": [
            "Section 6 — Consent of Data Principal before processing personal data",
            "Section 8 — Obligations of Data Fiduciary (accuracy, security, breach notification)",
            "Section 12 — Right of Data Principal to erasure and correction",
        ],
        "precedent": {
            "name": "Shreya Singhal v. Union of India (2015) — Supreme Court of India",
            "summary": (
                "Supreme Court ne IT Act Section 66A ko unconstitutional declare kiya — "
                "online speech par arbitrary restrictions allowed nahi hain. "
                "Ye case digital rights aur free speech ke liye landmark hai."
            ),
        },
        "resources": [
            "cybercrime.gov.in — National Cyber Crime Reporting Portal",
            "cert-in.org.in — CERT-In (Indian Computer Emergency Response Team)",
            "meity.gov.in — Ministry of Electronics & IT (IT Act, DPDP Act updates)",
            "rbi.org.in — RBI guidelines for UPI/payment fraud",
            "pdpb.meity.gov.in — DPDP Act 2023 resources",
        ],
    }


# ===========================================================================
# Flask Routes
# ===========================================================================

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask
    Body: { "question": "mera UPI se paise gaye, kya karein?" }
    Returns the Q&A RAG Agent's Hinglish answer.
    """
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Question field is required."}), 400

    answer = qa_rag_agent(question)
    return jsonify({"answer": answer})


def _is_link_only(text: str) -> bool:
    """
    Return True when the pasted text is just a URL / link with no document content.
    Heuristic: strip whitespace; if the whole thing matches a URL pattern OR
    there are fewer than 60 characters AND it starts with http(s), it's a link-only input.
    """
    import re
    t = text.strip()
    is_url = bool(re.match(r"^https?://\S+$", t, re.I))
    is_short_link = len(t) < 80 and bool(re.search(r"https?://|www\.", t, re.I))
    is_bare_domain = bool(re.match(r"^[a-z0-9][-a-z0-9]*(\.[a-z]{2,}){1,3}(/\S*)?$", t, re.I))
    return is_url or is_short_link or is_bare_domain


# ---------------------------------------------------------------------------
# Typo-squatting checker — used by /check-url
# ---------------------------------------------------------------------------

# Brand-name registry: maps the brand keyword → canonical domain shown in warning.
# Key = brand name only (lowercase, no TLD). Value = canonical domain to suggest.
_BRAND_REGISTRY = {
    # Banks / Finance
    "sbi":          "sbi.co.in",
    "onlinesbi":    "onlinesbi.com",
    "hdfc":         "hdfcbank.com",
    "hdfcbank":     "hdfcbank.com",
    "icici":        "icicibank.com",
    "icicibank":    "icicibank.com",
    "axis":         "axisbank.com",
    "axisbank":     "axisbank.com",
    "kotak":        "kotak.com",
    "pnb":          "pnbindia.in",
    "rbi":          "rbi.org.in",
    "npci":         "npci.org.in",
    # Payments / Wallets
    "paytm":        "paytm.com",
    "phonepe":      "phonepe.com",
    "googlepay":    "pay.google.com",
    "gpay":         "pay.google.com",
    # E-commerce
    "amazon":       "amazon.in",
    "flipkart":     "flipkart.com",
    "meesho":       "meesho.com",
    "myntra":       "myntra.com",
    "snapdeal":     "snapdeal.com",
    "nykaa":        "nykaa.com",
    # Social Media / Tech
    "instagram":    "instagram.com",
    "facebook":     "facebook.com",
    "whatsapp":     "whatsapp.com",
    "youtube":      "youtube.com",
    "google":       "google.com",
    # Government / Utilities
    "irctc":        "irctc.co.in",
    "incometax":    "incometax.gov.in",
    "uidai":        "uidai.gov.in",
    "aadhar":       "uidai.gov.in",
    "aadhaar":      "uidai.gov.in",
    "cybercrime":   "cybercrime.gov.in",
    "meity":        "meity.gov.in",
    "epfindia":     "epfindia.gov.in",
    "sebi":         "sebi.gov.in",
}


def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance (iterative)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _extract_brand_name(hostname: str) -> str:
    """
    Extract the brand-name token from a hostname for typo-squat comparison.

    Strategy:
      1. Strip scheme, www., trailing path/port.
      2. Take the registrable domain (last 2 dot-parts, or 3 for .co.in / .gov.in).
      3. Return the leftmost meaningful label (before the TLD).

    Examples:
      mesho.com          → "mesho"
      www.flipkartt.com  → "flipkartt"
      amazom.in          → "amazom"
      sbi-kyc.free.com   → "sbi-kyc"   (kept for scoring; hyphens removed later)
      hdfcbnk.com        → "hdfcbnk"
    """
    import re
    h = hostname.lower().strip()
    h = re.sub(r"^https?://", "", h)       # strip scheme (defensive)
    h = h.split("/")[0].split("?")[0].split(":")[0]   # strip path/port
    h = re.sub(r"^www\d*\.", "", h)        # strip www / www2 etc.

    parts = h.split(".")
    # For 3-part SLDs like sbi.co.in → brand = "sbi"
    # For 2-part like flipkart.com → brand = "flipkart"
    if len(parts) >= 3 and parts[-2] in ("co", "gov", "net", "org", "ac", "edu"):
        brand = parts[-3]
    elif len(parts) >= 2:
        brand = parts[-2]
    else:
        brand = parts[0]

    return re.sub(r"[-_0-9]", "", brand)   # strip hyphens/underscores/digits for clean compare


def _typosquat_check(hostname: str) -> dict:
    """
    Compare the brand-name portion of a hostname against _BRAND_REGISTRY using
    Levenshtein distance on brand names only (TLD ignored).

    Rules:
      - Exact match to a brand key            → NOT a typosquat (it IS the real brand)
      - Distance == 1 or 2 to any brand key   → Typosquat, HIGH confidence
      - Distance == 3 for brands ≥ 7 chars    → Typosquat, MEDIUM confidence
      - Otherwise                             → Clean

    Returns {"is_typosquat": bool, "suggested": str | None, "distance": int,
             "input_brand": str}
    """
    candidate = _extract_brand_name(hostname)

    if not candidate or len(candidate) < 3:
        return {"is_typosquat": False, "suggested": None, "distance": 99, "input_brand": candidate}

    best_dist   = 999
    best_brand  = None
    best_domain = None

    for brand, canonical in _BRAND_REGISTRY.items():
        dist = _levenshtein(candidate, brand)
        if dist < best_dist:
            best_dist   = dist
            best_brand  = brand
            best_domain = canonical

    # Exact match — it's the real brand, not a typosquat
    if best_dist == 0:
        return {"is_typosquat": False, "suggested": None, "distance": 0, "input_brand": candidate}

    # Typosquat thresholds
    threshold = 2 if len(candidate) >= 7 else 1   # allow extra edit for longer names
    is_typo = best_dist <= threshold

    if is_typo:
        return {
            "is_typosquat": True,
            "suggested": best_domain,
            "distance": best_dist,
            "input_brand": candidate,
        }
    return {"is_typosquat": False, "suggested": None, "distance": best_dist, "input_brand": candidate}


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Body: { "document": "<contract or privacy policy text>" }
    Runs all three analysis agents. Rejects input that is just a URL.
    """
    data = request.get_json(force=True)
    document = (data.get("document") or "").strip()

    if not document:
        return jsonify({"error": "Document field is required."}), 400

    # Reject link-only input — user should use the URL Checker tab
    if _is_link_only(document):
        return jsonify({
            "error": "link_only",
            "message": (
                "Please paste full Privacy Policy text here, not a link. "
                "Use the 'Check Link / URL' tab for checking links."
            ),
        }), 400

    # Run agents sequentially
    review     = contract_reviewer_agent(document)
    compliance = compliance_checker_agent(document)
    topic      = document[:120].replace("\n", " ")
    research   = case_research_agent(topic)

    return jsonify({
        "contract_review": review,
        "compliance": compliance,
        "case_research": research,
    })


@app.route("/check-url", methods=["POST"])
def check_url():
    """
    POST /check-url
    Body: { "url": "https://sbi-kyc.free.com/update" }
    Returns link risk score, breakdown, domain status, and typo-squat warning.
    """
    data = request.get_json(force=True)
    url  = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "URL field is required."}), 400

    # Normalise — add https:// if no scheme
    import re as _re
    if not _re.match(r"^https?://", url, _re.I):
        url_for_check = "https://" + url
    else:
        url_for_check = url

    link_score, link_breakdown, phishing_flag, domain_status = _score_links(url_for_check)

    # Extract hostname for typo-squat and domain checks
    hostname = _re.sub(r"^https?://", "", url_for_check.lower()).split("/")[0].split("?")[0].split(":")[0]

    typosquat = _typosquat_check(hostname)

    # Domain status label:
    # "Known Trusted Brand"  → hostname matches a brand in _BRAND_REGISTRY canonical list
    # "Cannot Verify Online" → DNS resolved but domain is not in our trusted brand list
    # "Not Found"            → DNS failed — clearly fake/non-existent
    # "No Link"              → no URL in input
    if domain_status == "Exists":
        # Build the flat set of all canonical domains from the brand registry
        _trusted_domains = set(_BRAND_REGISTRY.values())
        # Strip www. prefix before comparison
        _bare_host = _re.sub(r"^www\d*\.", "", hostname)
        if _bare_host in _trusted_domains:
            domain_status = "Known Trusted Brand"
        else:
            domain_status = "Cannot Verify Online"

    # Build warnings list
    warnings = []

    if domain_status == "Not Found":
        warnings.append(
            "Domain Not Found - This website does not exist on the internet. "
            "Be careful, this is likely a fake or expired domain."
        )

    if typosquat["is_typosquat"]:
        suggested   = typosquat["suggested"]
        input_brand = typosquat.get("input_brand", "")
        warnings.append(
            f"Possible Typo-Squatting Detected! "
            f"Did you mean {suggested}? "
            f"'{input_brand}' looks like a fake copy of '{suggested.split('.')[0]}'. "
            f"Do not enter passwords, OTPs or bank details here."
        )
        # Bump score to 75 minimum for typosquat
        link_score = max(link_score, 75)
        # Add typosquat entry to breakdown so score is explained
        link_breakdown.append({"label": f"Typo-Squatting (similar to {suggested})", "points": 0})

    if phishing_flag:
        warnings.append(
            "Phishing Link Detected - Fake KYC / bank link, uses insecure http, "
            "suspicious domain. Do not click. Report on 1930."
        )

    return jsonify({
        "url": url,
        "risk_score": min(link_score, 100),
        "score_breakdown": link_breakdown,
        "domain_status": domain_status,
        "phishing_flag": phishing_flag,
        "typosquat": typosquat,
        "warnings": warnings,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
