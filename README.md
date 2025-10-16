# Agentic-AI

Synthia Data Agent is an end-to-end Databricks solution for generating high-quality synthetic datasets through an AI agent workflow. The project demonstrates how to orchestrate multiple agents, Databricks Unity Catalog assets, and MLflow's ResponsesAgent interface to deliver an interactive synthetic data experience.

## Project Highlights
- **Agentic pipeline**: A planner agent (LangGraph + Databricks LLM) interprets user intent and prepares dataset specifications, handing them to downstream agents.
- **Databricks native**: Bundled resources provision pipelines, jobs, and Unity Catalog assets that run entirely on Databricks serverless or connected compute.
- **Governed data validation**: The data inspector agent validates catalogs, schemas, and tables via Unity Catalog SQL helper functions before profiling source data.
- **MLflow integration**: Agents are packaged as `ResponsesAgent` implementations, enabling evaluation, logging, and deployment from notebooks or Python code.
- **Playground-first development**: Notebooks exported from the Databricks AI Playground document experimentation, evaluation, and deployment workflows.

## Repository Layout
- `synthia_data_agent/`
  - `src/agents/`: Python agent implementations (currently the planner agent packaged for MLflow deployment).
  - `src/prompts/`: System prompts that define planner and data inspector behaviors.
  - `src/configs/`: Environment configuration, including the Databricks LLM endpoint name.
  - `src/playground/`: Auto-generated Databricks notebooks used to prototype and evaluate agents.
  - `resources/`: Databricks bundle definitions for pipelines and jobs.
  - `tests/`: Pytest suite scaffolding for Databricks Connect based validation.
  - `pyproject.toml`: Python project definition with `uv` build configuration.

## Prerequisites
- Python 3.10-3.13.
- [`uv`](https://docs.astral.sh/uv/) for dependency management and wheel builds.
- Access to a Databricks workspace with:
  - A Mosaic AI or Databricks Model Serving endpoint named in `src/configs/variables.py`.
  - Unity Catalog permissions to read source tables and publish synthetic outputs.
  - Databricks Bundle permissions to deploy pipelines and jobs specified in `databricks.yml`.

## Local Development
1. **Install dependencies**
   ```bash
   uv sync --group dev
   ```
   This installs pytest, Databricks LangChain integrations, and Databricks Connect for local execution.

2. **Configure environment**
   - Update catalog, schema, and LLM endpoint values in `synthia_data_agent/src/configs/variables.py`.
   - Ensure Databricks credentials are available (e.g., via `databricks configure` or environment variables) before running Databricks Connect or Bundles.

3. **Run tests**
   ```bash
   uv run pytest
   ```
   Tests assume connectivity to a Databricks workspace and a sample dataset accessible via Unity Catalog.

## Databricks Deployment
1. **Bundle validation and deployment**
   ```bash
   databricks bundle validate
   databricks bundle deploy --target dev
   ```
2. **Run the job**
   ```bash
   databricks bundle run --target dev synthia_data_agent_job
   ```
   The deployed job executes the exported notebooks, refreshes the Delta Live Tables pipeline, and runs the wheel entry point `synthia_data_agent.main:main`.

3. **Serving the planner agent**
   - The planner agent (`src/agents/planner_agent.py`) is registered with MLflow and exposed through Databricks Model Serving via the notebook in `src/playground/planner_agent_with_playground/driver.py`.
   - Use the MLflow UI to promote the model to Unity Catalog and deploy the serving endpoint referenced in `variables.py`.

## Extending the Agents
- Implement the data inspector agent by following the prompt contract in `src/prompts/data_inspector.py` and packaging it similarly to the planner agent.
- Add synthetic data generation tools (CTGAN, CopulaGAN, TVAE, etc.) and connect them as downstream nodes in the LangGraph state machine.
- Create Unity Catalog functions referenced in the prompts using the notebook in `src/tools/create_uc_function.ipynb`.

## Additional Notes
- The Databricks notebooks (`*.ipynb`) were exported from the UI and include `%magic` commands that run only inside Databricks.
- Encoding artifacts in some legacy files originate from the export process; normalize them if you plan to publish external documentation.
