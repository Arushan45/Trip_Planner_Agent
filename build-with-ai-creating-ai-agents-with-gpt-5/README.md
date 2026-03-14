# Build with AI: Creating AI Agents with GPT-5
This repository contains the Python code examples from the LinkedIn Learning course **Build with AI: Creating AI Agents with GPT-5**.

You will learn how to:
- Build agents that can call tools and take action
- Steer output using verbosity and reasoning settings
- Extend agents with custom tools for more capabilities

## Requirements
- Python 3.9+
- A Gemini API key
- A Weather API key (optional but recommended)

## Setup

1. **Clone this repo** (or download the files).
2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Set your Gemini API key or place in .env file**:
    ```bash
    export GEMINI_API_KEY="your_api_key"      # macOS/Linux
    setx GEMINI_API_KEY "your_api_key"        # Windows PowerShell
    ```
5. **Optional: set your Weather API key**:
    ```bash
    export WEATHER_API_KEY="your_api_key"     # macOS/Linux
    setx WEATHER_API_KEY "your_api_key"       # Windows PowerShell
    ```

## Running the Examples

Run any lesson script directly. For example:

```bash
python Lesson_2/agent.py
```

To run the FastAPI app used in Lesson 5:

```bash
cd Lesson_5
uvicorn agent:app --reload
```
