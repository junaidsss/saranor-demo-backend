from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# =========================
# SARANOR AI SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
PASTE THE FINAL SARANOR SYSTEM PROMPT HERE
"""

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message}
        ]
    )
    return {"reply": response.choices[0].message.content}
