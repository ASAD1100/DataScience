import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.yfinance import YFinanceTools
import tensorflow as tf
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()
# --- Load Bitcoin Prediction Model ---
@st.cache_resource
def load_bitcoin_model():
    return tf.keras.models.load_model("model_5_lstm.keras")  # adjust filename

btc_model = load_bitcoin_model()

# --- Streamlit Config ---
st.set_page_config(page_title="💹 FinTech AI Agent", layout="wide")

# --- App Header ---
st.title("💹 FinTech AI Agent")
st.markdown(
    """
    Analyze financial data, get analyst summaries, and search the web —  
    powered by **Groq + Agno AI** 🚀
    """
)

# --- API Key Setup ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key or groq_api_key == "YOUR_GROQ_KEY_HERE":
    st.warning("⚠️ No GROQ_API_KEY found. Please add it to your .env file.")

# --- Initialize Groq Model ---
groq_model = Groq(id="groq/compound", api_key=groq_api_key)

# --- Agents ---
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for financial info",
    model=groq_model,
    tools=[GoogleSearchTools()],
    instructions=["Always include sources"],
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    role="Analyze financial data and provide insights",
    model=groq_model,
    tools=[YFinanceTools()],
    instructions=["Use tables to display financial data"],
    markdown=True,
)

multi_ai_agent = Agent(
    tools=[web_search_agent, finance_agent],
    model=groq_model,
    instructions=[
        "Always include sources",
        "Use tables to display financial data",
        "Summarize data neatly with markdown headings like ## News, ## Analyst Ratings, ## Market Data",
    ],
    markdown=True,
)

st.set_page_config(
    page_title="💹 FinTech AI Agent",
    layout="wide",        # ✅ important for responsiveness
    initial_sidebar_state="collapsed"
)

# --- Query Input ---
st.markdown("### 🔎 Ask Your Question")
query = st.text_input(
    "💬 Example: Summarize Bitcoin analyst recommendations and the latest financial news."
)

# --- Run Button ---
if st.button("Run Analysis"):
    if not query.strip():
        st.warning("Please enter a financial query first.")
    else:
        with st.spinner("🔍 Running analysis..."):
            try:
                # Run the agent
                result = multi_ai_agent.run(query)

                # ✅ FIX: Extract plain text from RunOutput
                response_text = ""
                if hasattr(result, "output"):
                    response_text = result.output
                elif hasattr(result, "content"):
                    response_text = result.content
                else:
                    response_text = str(result)

                # --- Smart Formatting ---
                def format_response(text):
                    if not isinstance(text, str):
                        return str(text)
                    sections = re.split(r"(?=## )", text)
                    formatted = ""
                    for sec in sections:
                        if sec.strip():
                            formatted += f"\n\n{sec.strip()}\n\n---"
                    return formatted

                formatted_output = format_response(response_text)

                # --- Display Output ---
                st.markdown("## 🧠 AI Insights")
                st.markdown(formatted_output, unsafe_allow_html=True)

                if "table" in response_text.lower():
                    st.success("📊 Financial data successfully retrieved!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
st.markdown("## 💰 Bitcoin Price Prediction")
st.markdown(":money_mouth_face: `Predictions can be off by 1400 points`")
st.markdown(":warning: ASAD is not responsible for your financial losses and this is currently just a side project."
"\nPlease refer to it in real life when its predictions are within the realm of perfection"
"\nThank you for your support `YOROSHIKUUU`")

if st.button("Predict Bitcoin Trend"):
    with st.spinner("📈 Fetching BTC data and predicting..."):
        import matplotlib.dates as mdates

        # --- Fetch BTC data ---
        data = yf.download("BTC-USD", period="60d")
        close_prices = data['Close'].values
        dates = data.index

        window = 7  # adjust per your model
        X = np.array([close_prices[-window:]]).reshape((1, window, 1))
        predicted_price = btc_model.predict(X)[0][0]
        future_date = dates[-1] + pd.Timedelta(days=1)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0e1117")
        ax.plot(dates, close_prices, color="#00bfff", linewidth=2.5, label="Actual BTC Price")
        ax.scatter(future_date, predicted_price, color="red", s=80, label="Predicted Next Day", zorder=5)

        # --- Dark Theme Styling ---
        ax.set_facecolor("#0e1117")  # dark background
        fig.patch.set_facecolor("#0e1117")

        # Labels and title (white)
        ax.set_title("Bitcoin Price Prediction (Past 60 Days + Next Day Forecast)",
                     fontsize=13, color="white", pad=10)
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Price (USD)", color="white")

        # Axis ticks (white)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Format dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        fig.autofmt_xdate(rotation=45)

        # Legend styling
        legend = ax.legend(facecolor="#0e1117", edgecolor="none", labelcolor="white")
        for text in legend.get_texts():
            text.set_color("white")

        # Remove gridlines & borders
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Annotation for predicted price
        ax.text(future_date, predicted_price, f"${predicted_price:,.2f}",
                color="white", fontsize=11, ha='left', va='bottom', weight='bold')

        # --- Display in Streamlit ---
        st.pyplot(fig, use_container_width= True, width = 'stretch')

        st.success(f"📈 Predicted next BTC price: **${predicted_price:,.2f}**")

# --- Footer ---
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 0.9em; color: green;'>
        Built using Streamlit, Agno, and Groq APIs BY YOURS TRULY  `ASAD AMAN WANI`.
    </div>
    """,unsafe_allow_html=True,)
st.markdown(
    """
    <style>
    /* Make text & layout scale nicely on small screens */
    @media (max-width: 768px) {
        h1, h2, h3, h4, h5, h6, p, div, span {
            font-size: 90% !important;
        }
        .stButton>button {
            width: 100% !important;
        }
        .css-1d391kg {  /* Main content padding fix */
            padding-left: 10px;
            padding-right: 10px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1">
    """,
    unsafe_allow_html=True
)


    

