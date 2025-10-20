# Metadata Enrichment Agent

## Overview

The Metadata Enrichment Agent is a conversational AI agent designed to analyze tables that lack metadata and automatically propose descriptive metadata for tables and columns.

Its goal is to simplify and accelerate the metadata enrichment process inside an enterprise data catalog (for example, Unity Catalog), ensuring semantic coherence and high information quality.

The agent supports two operating modes:

- Interactive mode: a chatbot-style workflow where users (data stewards, domain owners) review, refine and approve proposed metadata.
- Automatic mode: a pipeline-oriented workflow for batch or scheduled enrichment jobs that run with minimal human intervention.

---

## Key objectives

1. Discover tables or columns that are missing metadata.
2. Profile data (column names, types, sample values, distributions) to collect signals for description.
3. Generate semantically coherent descriptions and metadata suggestions.
4. Allow users to validate or edit proposed metadata before publishing.
5. Persist approved metadata into the enterprise catalog.

---

## Usage modes

- Interactive mode
  - Use the agent as a conversational assistant to get descriptive suggestions and edits in real time.
  - Ideal for data stewards and domain owners who want to review proposals before publishing.

- Automatic mode
  - Integrate the agent into ETL/ELT pipelines for batch or recurring enrichment runs.
  - Ideal for large-scale metadata initiatives or for keeping the catalog continuously updated.

---

## Architecture and components

- Data profiling: lightweight profiling routines that determine column types, counts, distinct values, null statistics and representative examples.
- Generation module: an LLM-powered component (local model or served via Databricks/Model Serving) that produces human-readable descriptions and suggested tags/type names.
- Review interface: an API, notebook or UI that allows manual validation and editing of proposed metadata.
- Persistence: integration with Unity Catalog or a different enterprise catalog for saving approved metadata.

---

## Example flow

1. Scan the catalog to locate tables missing descriptive metadata.
2. Profile candidate tables to extract signals and examples.
3. Generate table and column descriptions and suggested metadata.
4. (Optional) Manual review and approval by a steward.
5. Write approved metadata back to the catalog.

---

## Installation & prerequisites (summary)

- Python 3.10+ (or an environment compatible with the project's agents).
- Access to the enterprise data catalog (Unity Catalog) with permissions to write metadata.
- An LLM endpoint or a local model available to generate descriptive text.

For detailed deployment instructions and helper notebooks, see the files included in the bundle.

---

## Advanced — operational guidance

- Run a pilot on a development copy of the catalog to validate suggestion quality before production rollout.
- Version prompts and model configurations for auditability and rollback.
- Apply data governance and legal review for metadata derived from sensitive datasets.

---

## Contacts & further information

If you need help with integration, deployment or policy questions, contact the project owners or open an issue in this repository.
