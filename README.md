# Agentic-AI

This repository contains two related Databricks-focused agent projects that demonstrate how to orchestrate multi-agent workflows, Databricks Asset Bundles, Unity Catalog validations, and MLflow/Model Serving for synthetic data generation and metadata enrichment.

Contents

- `synthia_data_agent/` — A multi-agent framework to plan, validate and generate synthetic datasets on Databricks. Implements a planner agent, prompts, Databricks asset bundles (pipelines & jobs), and notebooks to prepare and run the system.

- `metadata_enrichment_agent/` — A lightweight Databricks asset bundle scaffold that contains definitions to deploy jobs/pipelines focused on metadata enrichment tasks.

Why this repo exists

- Demonstrate agent orchestration (planner + data inspector) that translates natural language requests into dataset specifications.

- Show Databricks-first patterns: Asset Bundles, Unity Catalog helper functions, Delta Live Tables pipelines, and Model Serving integration.

- Provide examples and notebooks for rapid experimentation in the Databricks Playground and for packaging agents as MLflow Responses Agents.

Repository layout (high level)

- `synthia_data_agent/`

  - `databricks.yml` — bundle manifest for Databricks Asset Bundles.

  - `resources/` — pipeline and job definitions (YAML) for pipelines and jobs to run on Databricks.

  - `src/` — source code and notebooks:

    - `agents/` — Python agent implementations (e.g. `planner_agent.py`).

    - `prompts/` — prompt templates for planner and data inspector agents.

    - `configs/` — runtime configuration and variables used by the agents.

    - `playground/` — driver scripts and playground notebooks used to prototype and test agents locally/in-DB.

    - `setup/`, `tools/` — notebooks to prepare environment and register Unity Catalog helper functions.

  - `tests/` — pytest tests that assume Databricks connectivity (Databricks Connect / integration tests).

  - `pyproject.toml` — Python project configuration and dependencies.

- `metadata_enrichment_agent/`

  - A Databricks Asset Bundle scaffold (manifest, README) to deploy metadata enrichment jobs and pipelines.

Quickstart — local development (short)

1. Install dependencies and tooling

- Python 3.10–3.13
- uv (dependency manager used in the projects): pip install uv
- Databricks CLI (v0.205+) if you plan to work with Bundles locally

1. Install project dependencies

From the repository root run:

```powershell
uv sync
```

1. Configure Databricks credentials (PowerShell example)

```powershell
setx DATABRICKS_HOST "https://<workspace>.azuredatabricks.net"
setx DATABRICKS_TOKEN "<personal-access-token>"
```

1. Update configuration

Edit `synthia_data_agent/src/configs/variables.py` (catalog, schema, LLM endpoint name, etc.).

1. Run tests (integration-style; they expect a reachable Databricks environment)

```powershell
uv run pytest
```

Databricks deployment (short)

- Validate and deploy the asset bundle to a target environment (example uses the `uv run` tasks configured in the projects):

  ```powershell
  uv run databricks bundle validate --target dev
  uv run databricks bundle deploy --target dev
  uv run databricks bundle run --target dev synthia_data_agent_job
  ```

- Use the Databricks UI or MLflow Model Registry to manage and serve any packaged agents (planner agent is shown in the `src/playground` driver).

Important files and next steps

- `synthia_data_agent/src/agents/planner_agent.py` — planner agent implementation and entrypoint.

- `synthia_data_agent/src/prompts/` — prompt contracts (planner, data inspector).

- `synthia_data_agent/src/tools/create_uc_function.ipynb` — notebook to register Unity Catalog helper functions used by the agents.

- `synthia_data_agent/resources/` — pipeline & job YAML definitions used by the Databricks bundle.

- `metadata_enrichment_agent/resources/` — bundle definitions for metadata-enrichment workflows.

Notes and recommendations

- Notebooks include Databricks-specific `%` magics and may need cleaning to run outside Databricks.

- Tests are integration-style and can require Databricks Connect or a configured remote cluster. If you want to run unit tests locally, consider mocking Databricks and MLflow clients.

- Keep prompts and agent contracts versioned and reviewed for compliance before generating synthetic data from production sources.

Where to get more details

- Read the per-project READMEs for deeper setup instructions and examples:

  - `synthia_data_agent/README.md` — in-depth setup, architecture, and run instructions.

  - `metadata_enrichment_agent/README.md` — Databricks Asset Bundle usage for the metadata enrichment scaffold.

If you'd like, I can also:

- Add a small example script to show the planner agent flow locally (mocking the Databricks calls).

- Add a CONTRIBUTING.md with development conventions for this repo.

---
Last updated: 2025-10-20
