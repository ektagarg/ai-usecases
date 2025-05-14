import streamlit as st
import pandas as pd
import re
import asyncio
import openai
from io import BytesIO
import os

# Set your OpenAI key here
openai.api_key = st.secrets["OPENAI_API_KEY"]

client = openai.AsyncOpenAI(api_key=openai.api_key)

st.set_page_config(page_title="Misconduct Score Dashboard", layout="centered")
st.title("🚨 Rider Misconduct Scorer")

uploaded_file = st.file_uploader("Upload a CSV (must have a 'feedback' column)", type="csv")

# Better regex: only matches 0-10
def extract_score(text: str):
    match = re.search(r"\b(10|[0-9])\b", text)
    return int(match.group()) if match else None

async def get_score_async(feedback):
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a quality control AI for rider behavior.\n\n"
                        "Your job is to classify rider misconduct from customer feedback.\n\n"
                        "Return ONLY an integer score from 0 to 10. Do NOT explain.\n\n"
                        "Scoring scale:\n"
                        "0 = no issue (e.g., timely, polite, smooth delivery)\n"
                        "1-3 = minor issue (e.g., slightly late, not helpful, bad attitude)\n"
                        "4-6 = moderate issue (e.g., unprofessional, ignored instructions, didn't come to door)\n"
                        "7-9 = serious issue (e.g., very rude, refused to deliver, threw items)\n"
                        "10 = extreme issue (e.g., physical threat, verbal abuse, theft)\n\n"
                        "Examples:\n"
                        "- \"Delivery was smooth\" → 0\n"
                        "- \"Delivery boy was not polite\" → 2\n"
                        "- \"He refused to come upstairs\" → 5\n"
                        "- \"He shouted and used bad words\" → 8\n"
                        "- \"He threatened me\" → 10"
                    )
                },
                {"role": "user", "content": feedback},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        score = extract_score(content)
        if score is None:
            print(f"⚠️ Couldn't extract score from: {content}")
        return score
    except Exception as e:
        print(f"❌ Error for: {feedback[:30]}... → {e}")
        return None

async def process_all_feedback(feedbacks):
    tasks = [get_score_async(fb) for fb in feedbacks]
    return await asyncio.gather(*tasks)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "feedback" not in df.columns:
            st.error("CSV must contain a 'feedback' column.")
        else:
            # Clean feedback
            df["feedback"] = df["feedback"].astype(str)
            df = df[df["feedback"].str.strip().str.len() > 3]

            if st.button("Score Feedback"):
                with st.spinner("Scoring feedbacks... please wait ⏳"):
                    scores = asyncio.run(process_all_feedback(df["feedback"].tolist()))
                    df["misconduct_score"] = scores
                    st.success("✅ Done! See below:")
                    st.dataframe(df)

                    buffer = BytesIO()
                    df.to_csv(buffer, index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="📥 Download CSV with Scores",
                        data=buffer,
                        file_name="scored_feedbacks.csv",
                        mime="text/csv",
                    )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
