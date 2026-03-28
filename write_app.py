app_code = '''import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, VideoUnavailable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
import requests
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="vidQ", page_icon="🎬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
.brand { font-family: "Syne", sans-serif; font-size: 2rem; font-weight: 700; }
.brand span { color: #E24B4A; }
.tagline { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
.chat-msg-user { background: #E24B4A; color: white; border-radius: 12px 12px 2px 12px; padding: 0.75rem 1rem; margin: 0.5rem 0; max-width: 80%; margin-left: auto; }
.chat-msg-ai { background: #f0f0f0; border-radius: 12px 12px 12px 2px; padding: 0.75rem 1rem; margin: 0.5rem 0; max-width: 80%; }
.stButton > button { background: #E24B4A; color: white; border: none; border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 500; }
.stButton > button:hover { background: #c93c3b; color: white; }
</style>
""", unsafe_allow_html=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "google/gemma-3-4b-it:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
COOKIES_PATH = "cookies.txt"

def extract_video_id(url):
    match = re.search(r"(?:v=|/embed/|/shorts/|youtu\\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_transcript(video_id):
    ytt = YouTubeTranscriptApi()
    if os.path.exists(COOKIES_PATH):
        transcript_list = ytt.fetch(video_id, cookie_path=COOKIES_PATH)
    else:
        transcript_list = ytt.fetch(video_id)
    return " ".join(chunk.text for chunk in transcript_list)

def call_ai(system, user, history=None):
    if history is None:
        history = []
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://vidq.app",
        "X-Title": "vidQ YouTube Tutor"
    }
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]
    resp = requests.post(OPENROUTER_URL, headers=headers, json={
        "model": MODEL, "messages": messages, "max_tokens": 1500, "temperature": 0.7
    })
    if not resp.ok:
        err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        raise Exception(f"AI error: {err}")
    return resp.json()["choices"][0]["message"]["content"]

def build_vector_store(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    embeddings = FakeEmbeddings(size=128)
    return FAISS.from_texts(chunks, embeddings), chunks

def get_relevant_chunks(store, query, k=3):
    docs = store.similarity_search(query, k=k)
    return "\\n\\n".join(d.page_content for d in docs)

def parse_json(text):
    clean = re.sub(r"```json|```", "", text).strip()
    return json.loads(clean)

for key, default in {
    "transcript": None, "vector_store": None, "video_id": None,
    "chat_history": [], "flashcards": [], "quiz": [], "quiz_answered": {}, "score": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(\'<div class="brand">vid<span>Q</span></div>\', unsafe_allow_html=True)
st.markdown(\'<div class="tagline">YouTube AI Tutor — Q&A · Flashcards · Quiz</div>\', unsafe_allow_html=True)

url = st.text_input("Paste a YouTube URL", placeholder="https://youtube.com/watch?v=...")

if st.button("Analyse video"):
    if not url:
        st.error("Please paste a YouTube URL.")
    elif not OPENROUTER_API_KEY:
        st.error("API key not found. Check your .env file.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Could not find a valid YouTube video ID.")
        else:
            with st.spinner("Fetching transcript..."):
                try:
                    transcript = get_transcript(video_id)
                    store, chunks = build_vector_store(transcript)
                    st.session_state.transcript = transcript
                    st.session_state.vector_store = store
                    st.session_state.video_id = video_id
                    st.session_state.chat_history = []
                    st.session_state.flashcards = []
                    st.session_state.quiz = []
                    st.session_state.quiz_answered = {}
                    st.session_state.score = 0
                    st.success(f"Transcript loaded — {len(transcript.split())} words extracted.")
                except NoTranscriptFound:
                    st.error("No transcript found. Try a video with captions enabled.")
                except VideoUnavailable:
                    st.error("Video is unavailable or private.")
                except Exception as e:
                    st.error(f"Error: {e}")

if st.session_state.transcript:
    vid_id = st.session_state.video_id
    st.image(f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg", width=320)
    tab_qa, tab_fc, tab_quiz = st.tabs(["💬 Q&A Chat", "🃏 Flashcards", "🧠 Quiz"])

    with tab_qa:
        st.markdown("#### Ask anything about the video")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f\'<div class="chat-msg-user">{msg["content"]}</div>\', unsafe_allow_html=True)
            else:
                st.markdown(f\'<div class="chat-msg-ai">{msg["content"]}</div>\', unsafe_allow_html=True)
        question = st.text_input("Your question", placeholder="What is the main topic?", key="qa_input")
        if st.button("Ask"):
            if question:
                with st.spinner("Thinking..."):
                    try:
                        context = get_relevant_chunks(st.session_state.vector_store, question)
                        system = f"You are an expert tutor. Answer based ONLY on this transcript context.\\n\\nCONTEXT:\\n{context}"
                        history = [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]} for m in st.session_state.chat_history[-6:]]
                        answer = call_ai(system, question, history)
                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_fc:
        st.markdown("#### Study flashcards")
        if st.button("Generate flashcards"):
            with st.spinner("Generating..."):
                try:
                    result = call_ai(
                        \'Generate exactly 6 flashcards. Return ONLY a raw JSON array, no markdown:\\n[{"front":"question","back":"answer"},...]\',
                        f"TRANSCRIPT:\\n{st.session_state.transcript[:3000]}"
                    )
                    st.session_state.flashcards = parse_json(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        if st.session_state.flashcards:
            st.caption("Click to expand and see the answer")
            for i, card in enumerate(st.session_state.flashcards):
                with st.expander(f"Card {i+1}: {card[\'front\']}"):
                    st.markdown(f"**Answer:** {card[\'back\']}")

    with tab_quiz:
        st.markdown("#### Test your understanding")
        if st.button("Generate quiz"):
            with st.spinner("Generating..."):
                try:
                    result = call_ai(
                        \'Generate exactly 5 multiple-choice questions. Return ONLY a raw JSON array, no markdown:\\n[{"question":"...","options":["A","B","C","D"],"correct":0},...]\',
                        f"TRANSCRIPT:\\n{st.session_state.transcript[:3000]}"
                    )
                    st.session_state.quiz = parse_json(result)
                    st.session_state.quiz_answered = {}
                    st.session_state.score = 0
                except Exception as e:
                    st.error(f"Error: {e}")
        if st.session_state.quiz:
            correct_count = sum(1 for k, v in st.session_state.quiz_answered.items() if st.session_state.quiz[k]["correct"] == v)
            total = len(st.session_state.quiz)
            answered = len(st.session_state.quiz_answered)
            st.progress(answered / total if total else 0)
            st.markdown(f"**Score: {correct_count}/{total}**")
            for i, q in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}. {q[\'question\']}**")
                answered_this = i in st.session_state.quiz_answered
                for j, opt in enumerate(q["options"]):
                    label = opt
                    if answered_this:
                        if j == q["correct"]: label = "✅ " + opt
                        elif st.session_state.quiz_answered[i] == j: label = "❌ " + opt
                        st.button(label, key=f"q{i}_o{j}", disabled=True)
                    else:
                        if st.button(label, key=f"q{i}_o{j}"):
                            st.session_state.quiz_answered[i] = j
                            st.rerun()
                if answered_this:
                    if st.session_state.quiz_answered[i] == q["correct"]:
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect — correct answer: {q[\'options\'][q[\'correct\']]}")
                st.divider()
            if answered == total:
                pct = round((correct_count / total) * 100)
                if correct_count == total:
                    st.balloons()
                    st.success(f"🏆 Perfect! {correct_count}/{total} — {pct}%")
                elif correct_count >= 3:
                    st.success(f"👍 Good job! {correct_count}/{total} — {pct}%")
                else:
                    st.warning(f"📚 Keep studying! {correct_count}/{total} — {pct}%")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("app.py created successfully!")
