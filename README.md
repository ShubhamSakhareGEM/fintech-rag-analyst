# FinSight: Financial 10-K Report Analyst

**Live Demo:** [Click here to use FinSight](https://fintech-rag-analyst-3xpu7schfrwwsfbzrn9ip4.streamlit.app/)

<img width="1440" height="861" alt="Screenshot 2025-12-30 at 14 22 02" src="https://github.com/user-attachments/assets/2207bd48-3609-4ecc-87d4-f82947ee4a1e" />

I built **FinSight** which is a Retrieval-Augmented Generation (RAG) application designed to assist financial analysts and investors. It allows users to upload complex financial documents (like SEC 10-K filings, Earnings Call transcripts, or Insurance Policies) and chat with them to extract specific metrics, risk factors, and summaries.

I worked on this project to move beyond basic CRUD applications and explore **GenAI engineering**, specifically focusing on how to handle large context windows and vector retrieval in a financial context.

-----

## Tech Stack

  * **Language:** Python 3.10+
  * **Frontend:** Streamlit
  * **Orchestration:** LangChain
  * **LLM (Inference):** Google Gemini (Model: `gemini-2.5-flash`)
  * **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) - *Running locally for privacy & cost efficiency*
  * **Vector Database:** FAISS (Facebook AI Similarity Search)
  * **PDF Processing:** PyPDF2

-----

## How to Run Locally

Follow these steps to set up the project on your machine (Mac/Windows/Linux).

### 1\. Clone the Repository

```bash
git clone https://github.com/yourusername/fintech-rag-analyst.git
cd fintech-rag-analyst
```

### 2\. Create a Virtual Environment (Recommended)

This keeps your system clean and avoids dependency conflicts.

```bash
# MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Get a Free Google API Key

This project uses Google's Gemini model for the reasoning engine.

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Log in with your Google account.
3.  Click **"Get API key"** -\> **"Create API key in new project"**.
4.  Copy the key string (starts with `AIza...`). You will need to paste this into the app's sidebar.

### 5\. Run the Application

```bash
streamlit run app.py
```

A browser window should open automatically at `http://localhost:8501`.

-----

## Where to Get Sample Financial Data

You can download real 10-K filings (Annual Reports) from any public company's "Investor Relations" page. Here are direct links to good examples:

1.  **Apple (AAPL) 10-K:** [Download PDF](https://investor.apple.com/sec-filings/default.aspx) (Look for "Annual Report on Form 10-K")
2.  **Tesla (TSLA) 10-K:** [Download PDF](https://ir.tesla.com/sec-filings)
3.  **Microsoft (MSFT) 10-K:** [Download PDF](https://www.microsoft.com/en-us/investor/sec-filings.aspx)

**Note:** The app works best with standard text-based PDFs. Scanned images (OCR) are not currently supported in this version.

-----

## Sample Questions to Ask

Once you upload a 10-K report, try these prompts:

  * **Summarization:** "Summarize the 'Management's Discussion and Analysis' section in 3 bullet points."
  * **Risk Analysis:** "What are the top 3 risk factors mentioned regarding AI or cybersecurity?"
  * **Data Extraction:** "What was the total net revenue for 2023, 2024, and 2025?"
  * **Legal:** "Are there any pending lawsuits mentioned in the 'Legal Proceedings' section?"

-----

## Technical Challenges & Solutions

While building this, I encountered real-world engineering hurdles:

**1. API Rate Limiting (The "429" Error):**

  * **Issue:** Initially, I used Google's API for *both* embeddings and chat. Processing a 100-page PDF triggered Google's free-tier rate limits immediately.
  * **Solution:** I decoupled the architecture. I switched to **HuggingFace (`all-MiniLM-L6-v2`)** for embeddings. This runs locally on the CPU, is 100% free, and saves the API quota strictly for the high-value reasoning (LLM) tasks.

**2. Model Deprecation:**

  * **Issue:** The initial tutorial code used `gemini-pro`, which Google deprecated.
  * **Solution:** I upgraded the codebase to use `gemini-2.5-flash`, ensuring the app uses the latest, most cost-effective model available in late 2025.

**3. Dependency Management:**

  * **Issue:** LangChain updates frequently, causing `ModuleNotFoundError`.
  * **Solution:** I refactored the code to use the modern `langchain-community` and `langchain-google-genai` libraries and unpinned specific versions to allow for compatibility fixes on different OS architectures (M1 Mac vs Intel).

**4. The "SQLite Version Mismatch" Crash**

  * **Issue:** The default Linux environment on Streamlit Cloud uses an older version of SQLite (v3.31), but the vector database (Chroma/FAISS) requires a newer version (v3.35+). This caused the app to crash immediately upon deployment.
  * **Solution:** I implemented a "hot-swap" patch in app.py. By installing pysqlite3-binary and injecting it into sys.modules at runtime, I forced the application to use a modern, compatible database engine without needing root server access.

**5. Python 3.13 Compatibility**

  * **Issue:** Streamlit Cloud defaults to the latest Python 3.13. However, critical libraries like langchain-community and faiss-cpu have not yet released stable wheels for 3.13, leading to ModuleNotFoundError: No module named 'langchain.chains'.
  * **Solution:** I explicitly pinned the environment to Python 3.10 in the deployment settings and locked the requirements.txt versions (langchain==0.1.16) to ensure a stable, reproducible build.

-----

## Future Improvements

If I were to take this to production, I would add:

1.  **Citations:** make the bot cite exactly which page number the answer came from.
2.  **Chat History:** Allow the user to ask follow-up questions (currently it treats every question as new).
3.  **Table Parsing:** Improve the ability to read complex financial tables using a tool like *LlamaParse*.
4.  **Dockerization:** Containerize the app for easier deployment on AWS/GCP.
