import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

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

# --- Footer ---
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        Built with ❤️ using Streamlit, Agno, and Groq APIs.
    </div>
    """,
    unsafe_allow_html=True,
)
