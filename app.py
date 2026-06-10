from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL, GEMINI_API_KEY
from evaluator import EvaluationRunner
from metrics import summarize_prompt_metrics
from storage import PromptStorage, RunStorage
from utils import load_csv, validate_csv_file


def show_header() -> None:
    st.title("Persian Prompt Evaluation Studio")
    st.markdown(
        "A Streamlit-based workflow for prompt engineering, Persian NLP evaluation, and LLM scoring analytics."
    )
    if not GEMINI_API_KEY:
        st.error(
            "Missing Gemini API key. Set `GEMINI_API_KEY` in your environment variables or `.env` file."
        )
        st.stop()


def show_home(prompts: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    st.header("Home")
    st.write(
        "Use this studio to upload Persian datasets, manage prompt templates, execute multi-prompt evaluations, and compare results through dashboards."
    )

    col1, col2 = st.columns(2)
    col1.metric("Saved Prompts", len(prompts), help="Number of prompt templates stored locally.")
    col2.metric("Recorded Runs", len(runs), help="Total evaluation experiments saved.")

    if runs:
        last_run = runs[0]
        st.subheader("Latest Run")
        st.markdown(f"**Dataset:** {last_run.get('dataset_name', 'Unknown')}  ")
        st.markdown(f"**Timestamp:** {last_run.get('timestamp')}  ")
        st.markdown(f"**Prompt Count:** {len(last_run.get('prompts', []))}")
        st.metric("Average overall score", f"{last_run['metrics']['average_score']:.2f}")
        st.metric("Average latency", f"{last_run['metrics']['average_latency']:.2f}s")

    st.info(
        "This application is designed for Persian prompt evaluation and integrates automated judge scoring, cost estimation, and prompt comparison analytics."
    )


def show_prompt_management(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    st.header("Prompt Management")
    st.write("Create, edit, delete, and save prompt templates for your Persian NLP evaluation experiments.")

    saved_names = [prompt["name"] for prompt in prompts]
    selected_prompt = st.selectbox("Select prompt to edit", ["Create new prompt"] + saved_names)

    if selected_prompt != "Create new prompt":
        prompt_data = next(item for item in prompts if item["name"] == selected_prompt)
    else:
        prompt_data = {"name": "", "description": "", "template": "You are an expert assistant.\n\nAnswer the following question:\n\n{input}"}

    name = st.text_input("Prompt name", prompt_data["name"])
    description = st.text_area("Description", prompt_data["description"], height=100)
    template = st.text_area("Prompt template", prompt_data["template"], height=220)

    col1, col2 = st.columns(2)
    if col1.button("Save prompt"):
        if not name.strip() or not template.strip():
            st.warning("Both name and prompt template are required.")
        else:
            prompt_record = {"name": name.strip(), "description": description.strip(), "template": template.strip()}
            PromptStorage.save_prompt(prompt_record)
            st.success(f"Prompt '{name}' saved.")

    if selected_prompt != "Create new prompt" and col2.button("Delete prompt"):
        PromptStorage.delete_prompt(selected_prompt)
        st.success(f"Prompt '{selected_prompt}' deleted.")

    if prompts:
        st.divider()
        st.subheader("Saved prompts")
        for prompt in prompts:
            with st.expander(prompt["name"]):
                st.markdown(f"**Description:** {prompt['description']}")
                st.code(prompt["template"], language="text")

    return PromptStorage.load_prompts()


def show_run_evaluation(prompts: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    st.header("Run Evaluation")
    st.write("Upload a Persian CSV dataset, select an input column, and run multiple prompts against all rows.")

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    dataset_name = "Uploaded dataset"
    selected_column = None
    dataframe: pd.DataFrame | None = None

    if uploaded_file is not None:
        validation = validate_csv_file(uploaded_file)
        if validation["valid"]:
            dataframe = load_csv(uploaded_file)
            if dataframe.empty:
                st.error("The dataset is empty. Please upload a valid CSV file.")
                return
            selected_column = st.selectbox("Select input text column", dataframe.columns.tolist())
            dataset_name = uploaded_file.name
            st.write(f"Dataset loaded with {len(dataframe)} rows.")
        else:
            st.error(validation["error"])
            return

    if not prompts:
        st.warning("Add at least one prompt in Prompt Management before running an evaluation.")
        return

    model = st.selectbox("Generation model", [DEFAULT_MODEL], index=0)
    judge_model = st.selectbox("Judge model", [DEFAULT_JUDGE_MODEL], index=0)
    max_rows = st.number_input("Maximum rows to evaluate", min_value=1, value=20, step=1)
    use_cost_estimate = st.checkbox("Show estimated token usage and cost", value=True)
    if use_cost_estimate:
        st.info("Token usage and estimated cost are collected for each evaluation row.")

    if st.button("Run evaluation"):
        if dataframe is None or selected_column is None:
            st.warning("Please upload a dataset and select a column before running an evaluation.")
            return

        inputs = [str(row) for row in dataframe[selected_column].fillna("")[:max_rows].tolist()]
        prompts_map = prompts

        evaluation_runner = EvaluationRunner(model=model, judge_model=judge_model)
        with st.spinner("Running evaluation for all prompts..."):
            run_data = evaluation_runner.run(
                dataset_name=dataset_name,
                dataset_inputs=inputs,
                prompts=prompts_map,
            )

        RunStorage.save_run(run_data)
        st.success("Evaluation complete. Run saved.")

    if runs:
        st.divider()
        st.subheader("Recent experiment summaries")
        for run in runs[:3]:
            metrics = run["metrics"]
            st.markdown(f"**{run['timestamp']}** — {run['dataset_name']}")
            st.write(
                f"Prompts: {len(run['prompts'])} | Total evaluations: {metrics['total_evaluations']} | "
                f"Avg. score: {metrics['average_score']:.2f} | Avg. latency: {metrics['average_latency']:.2f}s"
            )


def show_results(runs: list[dict[str, Any]]) -> None:
    st.header("Results")
    st.write("Browse previous evaluation runs, inspect response data, and export experiment results.")

    if not runs:
        st.info("No saved runs available yet. Run an evaluation to generate results.")
        return

    run_labels = [f"{run['timestamp']} - {run['dataset_name']}" for run in runs]
    selected_index = st.selectbox("Select saved run", list(range(len(runs))), format_func=lambda x: run_labels[x])
    selected_run = runs[selected_index]

    st.markdown(f"### {selected_run['timestamp']} — {selected_run['dataset_name']}")
    metrics = selected_run["metrics"]
    st.metric("Average overall score", f"{metrics['average_score']:.2f}")
    st.metric("Average latency", f"{metrics['average_latency']:.2f}s")
    st.metric("Total evaluations", metrics["total_evaluations"])
    st.metric("Total tokens", metrics.get("total_tokens", 0))
    st.metric("Estimated cost", f"${metrics.get('total_estimated_cost', 0.0):.6f}")

    row_data = []
    for item in selected_run["rows"]:
        row_data.append(
            {
                "input": item["input_text"],
                "prompt": item["prompt_name"],
                "response": item["response_text"],
                "accuracy": item["judge_score"]["accuracy"],
                "completeness": item["judge_score"]["completeness"],
                "clarity": item["judge_score"]["clarity"],
                "relevance": item["judge_score"]["relevance"],
                "overall": item["judge_score"]["overall"],
                "latency": item["latency_seconds"],
            }
        )

    st.dataframe(pd.DataFrame(row_data))

    with st.expander("Export current run"):
        run_json = json.dumps(selected_run, ensure_ascii=False, indent=2)
        st.download_button("Download JSON", run_json, file_name=f"run_{selected_run['timestamp']}.json")
        result_df = pd.DataFrame(row_data)
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv_bytes, file_name=f"run_{selected_run['timestamp']}.csv")


def show_analytics(runs: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> None:
    st.header("Analytics")
    st.write("Visualize prompt performance, latency, score distributions, and ranking analytics.")

    if not runs:
        st.info("No runs available yet. Run an evaluation to see analytics.")
        return

    selected_run = st.selectbox("Choose a run for analytics", runs, format_func=lambda r: f"{r['timestamp']} - {r['dataset_name']}")
    prompt_metrics = summarize_prompt_metrics(selected_run)

    if prompt_metrics.empty:
        st.warning("Selected run does not contain enough data for analytics.")
        return

    fig_score = px.bar(
        prompt_metrics,
        x="prompt_name",
        y="average_overall",
        title="Average overall score by prompt",
        labels={"prompt_name": "Prompt", "average_overall": "Average Score"},
    )
    st.plotly_chart(fig_score, use_container_width=True)

    fig_latency = px.bar(
        prompt_metrics,
        x="prompt_name",
        y="average_latency",
        title="Average latency by prompt",
        labels={"prompt_name": "Prompt", "average_latency": "Latency (s)"},
    )
    st.plotly_chart(fig_latency, use_container_width=True)

    distribution_df = pd.DataFrame(
        [
            {
                "prompt_name": item["prompt_name"],
                "overall": item["judge_score"]["overall"],
            }
            for item in selected_run["rows"]
        ]
    )
    fig_distribution = px.box(
        distribution_df,
        x="prompt_name",
        y="overall",
        title="Individual prompt score distribution",
        labels={"prompt_name": "Prompt", "overall": "Overall score"},
    )
    st.plotly_chart(fig_distribution, use_container_width=True)

    st.markdown("### Prompt ranking")
    st.dataframe(prompt_metrics.sort_values(by="average_overall", ascending=False))

    if st.checkbox("Show raw aggregate metrics"):
        st.json(selected_run["metrics"])


def main() -> None:
    st.set_page_config(
        page_title="Persian Prompt Evaluation Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    PromptStorage.initialize()
    RunStorage.initialize()

    show_header()
    prompts = PromptStorage.load_prompts()
    runs = RunStorage.load_runs()

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Prompt Management", "Run Evaluation", "Results", "Analytics"],
    )

    if page == "Home":
        show_home(prompts, runs)
    elif page == "Prompt Management":
        prompts = show_prompt_management(prompts)
    elif page == "Run Evaluation":
        show_run_evaluation(prompts, runs)
    elif page == "Results":
        show_results(runs)
    elif page == "Analytics":
        show_analytics(runs, prompts)


if __name__ == "__main__":
    main()
