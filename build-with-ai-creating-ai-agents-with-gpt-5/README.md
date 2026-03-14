# Build with AI: Creating AI Agents with GPT‑5
This repository contains the Python code examples from the LinkedIn Learning course **Build with AI: Creating AI Agents with GPT‑5**.

![Build with AI: Creating AI Agents with GPT-5](https://github.com/keshawillz/build-with-ai-creating-ai-agents-with-gpt-5/blob/main/course_image2.png)

You’ll learn how to:
- Build agents that can call tools and take action
- Steer GPT-5 output using verbosity and reasoning settings
- Extend agents with custom tools for more capabilities

## Requirements
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- A [Weather API key](https://www.weatherapi.com/my/)

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
4. **Set your OpenAI API key or place in .env file**:
    ```bash
    export OPENAI_API_KEY="your_api_key"      # macOS/Linux
    setx OPENAI_API_KEY "your_api_key"        # Windows PowerShell
    ```
5. **Set your Weather API key or place in .env file**:
    ```bash
    export WEATHER_API_KEY="your_api_key"      # macOS/Linux
    setx OPENAI_API_KEY "your_api_key"        # Windows PowerShell
    ```

## Running the Examples

Run the main demo script to see all lessons in action:

```bash
python agent.py

