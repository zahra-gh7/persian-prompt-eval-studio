# Persian Prompt Evaluation Studio

A Streamlit-based framework for evaluating Persian prompts, benchmarking LLM outputs, and analyzing prompt engineering experiments.

## Overview

Persian Prompt Evaluation Studio is a portfolio-ready AI/NLP project for prompt engineers and data practitioners working with Persian text.
It enables experiment design, multi-prompt evaluation, automated judge scoring, and analytics for prompt performance and generation quality.

## Key Highlights

- Built an end-to-end prompt evaluation workflow for Persian NLP tasks
- Integrated LLM-based automated judging and scoring
- Designed experiment tracking and prompt comparison pipelines
- Developed an interactive Streamlit dashboard for AI evaluation workflows

## Features

- Persian prompt management UI with create, edit, and delete functionality
- Upload and validate Persian CSV datasets
- Multi-prompt evaluation over dataset rows with Gemini generation
- Automated judge scoring for accuracy, completeness, clarity, relevance, and overall quality
- Prompt comparison dashboards, ranking, and exportable results
- Estimated token usage, modeled cost, and runtime metrics
- Persisted run history and experiment tracking

## Architecture

- `app.py` — Streamlit application, page navigation, and user workflows
- `config.py` — environment loading, path configuration, and default model settings
- `llm_client.py` — Gemini API integration and response handling
- `evaluator.py` — prompt evaluation workflow and result assembly
- `judge.py` — automated scoring prompt generation and judge response parsing
- `metrics.py` — aggregation, scoring, and prompt ranking logic
- `storage.py` — persistent prompt and run storage
- `utils.py` — CSV loading, validation, prompt rendering, and cost estimation

## Tech Stack

- Python 3.11+
- Streamlit
- pandas
- Plotly
- Google Gemini generative AI SDK
- python-dotenv

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd prompt-eval-studio
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the example:

```bash
cp .env.example .env
```

5. Add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your-gemini-api-key
```

## Configuration

Update `.env` to configure Gemini API settings.
The default model values are:

- `GEMINI_MODEL=gemini-2.0-flash`
- `GEMINI_JUDGE_MODEL=gemini-2.0-flash`

You can override token limits and temperature in `.env` as needed.

## Usage

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open the displayed local URL in your browser.

## Example Workflow

1. Open **Prompt Management** and add Persian prompt templates.
2. Upload a sample Persian dataset in CSV format.
3. Select the input text column and choose generation/judge models.
4. Run the evaluation across prompts and review saved run results.
5. Explore analytics, download JSON/CSV exports, and compare prompt performance.

## Project Structure

- `app.py` — Streamlit app and application flow
- `config.py` — config loading and default environment values
- `data/` — sample dataset assets
- `docs/screenshots/` — screenshot placeholders for documentation
- `evaluator.py` — prompt evaluation logic
- `judge.py` — judge scoring workflow
- `llm_client.py` — Gemini API adapter
- `metrics.py` — aggregation and scoring utilities
- `prompts/` — prompt templates and storage
- `results/` — generated experiment history
- `storage.py` — persistence utilities
- `utils.py` — helper functions and validation

## Sample Dataset

A sample dataset is available at `data/sample_questions.csv`.
Use this file to test Persian prompt evaluation workflows in the app.


## Future Improvements

- Add ground truth validation and reference-based scoring
- Support asynchronous evaluation and batch Gemini requests
- Add user accounts and project-level storage
- Add prompt versioning and collaboration features
- Support additional models and local open-source backends

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
