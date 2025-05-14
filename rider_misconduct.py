import streamlit as st
import pandas as pd
import openai
import re

from openai import OpenAI

# Set API key (from Streamlit secrets or directly)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Define prompt
default_prompt = """
You are an AI assistant tasked with analyzing customer feedback messages related to delivery services. Your objective is to categorize each message into one of the following levels of urgency:
Critical: Requires immediate attention due to urgent issues like safety concerns, significant financial discrepancies, or severe misconduct.
Neutral: Needs attention but is not immediately critical. This includes issues like late deliveries, minor order discrepancies, or general complaints.
Non-Critical: Does not require action, such as unrelated messages, thank you notes, or general inquiries not related to delivery partner behavior.

Additionally, identify the primary issue category (bucket) for each message from the following list:
Rude Behavior: Includes complaints about rude, unprofessional, or abusive language/actions from the delivery partner.
Delivery Issues: Concerns related to late delivery, wrong delivery, undelivered items, damaged goods, or issues with the delivery process.
Payment/Charges: Issues related to overcharging, incorrect payments, refund requests, or payment disputes.
Service Quality: General complaints about poor service, lack of communication, or unsatisfactory experience.
Safety Concern: Issues related to unsafe behavior by the delivery partner, or any situation that posed a safety risk.
Other: Issues that do not fit into the above categories.

You must respond in exactly this format:

Classification: <Critical / Neutral / Non-Critical>  
Bucket: <Rude Behavior / Delivery Issues / Payment/Charges / Service Quality / Safety Concern / Other>  
Justification: <One sentence>  
Score: <0 to 10>

Respond in the format exactly as shown above. Do not add any extra text.

"""

# Function to call GPT (sync version, for simplicity)
def analyze_feedback(feedback, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": feedback}
            ],
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()

        # Extract structured info
        classification = re.search(r"Classification\s*[:\-–]\s*(.+)", reply, re.IGNORECASE)
        bucket = re.search(r"Bucket\s*[:\-–]\s*(.+)", reply, re.IGNORECASE)
        justification = re.search(r"Justification\s*[:\-–]\s*(.+)", reply, re.IGNORECASE)
        score = re.search(r"Score\s*[:\-–]\s*(\d+)", reply, re.IGNORECASE)


        return (
            classification.group(1).strip() if classification else None,
            bucket.group(1).strip() if bucket else None,
            justification.group(1).strip() if justification else None,
            int(score.group(1)) if score else None,
        )

    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return None, None, None, None

# --- Streamlit UI ---
st.title("📊 Delivery Feedback Classifier & Scorer")

system_prompt = st.text_area("System Prompt", default_prompt, height=350)
uploaded_file = st.file_uploader("Upload a CSV file with a 'feedback' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "feedback" not in df.columns:
        st.error("CSV must have a 'feedback' column.")
    else:
        if st.button("Analyze"):
            with st.spinner("Processing..."):
                results = [analyze_feedback(fb, system_prompt) for fb in df["feedback"]]
                classifications, buckets, justifications, scores = zip(*results)

                df["Classification"] = classifications
                df["Bucket"] = buckets
                df["Justification"] = justifications
                df["Score"] = scores

                st.success("Done!")
                st.dataframe(df)
                st.download_button("Download results", df.to_csv(index=False), "classified_feedback.csv")
