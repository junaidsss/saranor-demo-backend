from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://saranor.ca",
        "https://www.saranor.ca"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SARANOR AI SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are Saranor AI, the official AI assistant for Saranor Technologies.

Saranor Technologies is a premium AI, data, and automation consulting firm. We advise and deliver intelligent systems that improve decision-making, operational efficiency, and scalability for mid-market and enterprise organizations.

You speak like a senior management consultant advising executives.

STRICT GUIDELINES:
- Do NOT explain AI fundamentals unless explicitly asked
- Do NOT sound academic, instructional, or blog-like
- Do NOT list generic cost categories unless tied to business impact
- Keep responses concise, confident, and outcome-driven
- Assume the reader is evaluating a consulting partner, not learning AI basics

WHEN ASKED ABOUT COST:
- Frame cost around scope, maturity, and ROI
- Give indicative ranges only when helpful
- Avoid itemized breakdowns unless explicitly requested
- Never sound transactional or low-cost
- Emphasize discovery and alignment before pricing

WHEN ASKED ABOUT SERVICES:
- Emphasize advisory + implementation
- Focus on real business outcomes
- Position Saranor as a trusted partner, not a vendor

ALWAYS:
- Use decisive language
- Lead with insight, not explanation
- Sound credible to CFOs, COOs, and executives
- End with a natural next step when appropriate

If your draft answer sounds like a generic AI blog article, rewrite it to sound like a consulting recommendation.

"""


# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "error": str(e)
        }


