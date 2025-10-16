# Synthia Data Agent

## Panoramica
Synthia Data Agent e un framework multi agente pensato per orchestrare la generazione di dataset sintetici direttamente su Databricks. Il sistema traduce richieste in linguaggio naturale, valida le sorgenti in Unity Catalog e produce specifiche di dataset che possono essere passate a modelli generativi come CTGAN, CopulaGAN o TVAE, mantenendo privacy e conformita.

## Caratteristiche principali
- Comprensione automatica delle richieste utente e pianificazione del dataset target.
- Validazione granulare di cataloghi, schemi e tabelle in Unity Catalog tramite funzioni SQL dedicate.
- Integrazione con LangChain, LangGraph e Databricks Model Serving per orchestrare agenti conversazionali.
- Tracciamento con MLflow Responses Agents per salvare cronologia e output dei nodi.
- Notebook e strumenti per configurare l'ambiente (funzioni SQL, setup del workspace, playground interattivo).

## Architettura logica
- Planner Agent (`src/agents/planner_agent.py`): riceve i messaggi, applica il prompt di `src/prompts/planner.py` e costruisce la richiesta strutturata tramite LangGraph.
- Data Inspector Agent: definito nel prompt `src/prompts/data_inspector.py`, verifica l'esistenza delle risorse Unity Catalog e raccoglie metadati della tabella.
- Funzioni SQL di supporto (`src/tools/create_uc_function.ipynb`): registra le funzioni `check_catalog_exist`, `check_schema_exist` e `check_table_exist` nello schema `agentic_ai.synthia_data_agent`.
- Pipeline e job Databricks: descritti in `resources/synthia_data_agent.pipeline.yml` e `resources/synthia_data_agent.job.yml`, orchestrati tramite Databricks Asset Bundles (`databricks.yml`).
- Notebook operativi (`src/1.planner_agent.ipynb`, `src/2.data_inspector_agent.ipynb`, `src/setup/prepare_environment.ipynb`) per esplorare, validare e preparare l'ambiente.

## Prerequisiti
- Workspace Databricks con Unity Catalog abilitato e accesso a Serverless SQL o cluster compatibili.
- Endpoint LLM su Databricks Model Serving (per esempio `databricks-llama-4-maverick`), configurato per accettare chiamate da LangChain.
- Python 3.10 - 3.13 e strumenti da riga di comando: Git, [uv](https://docs.astral.sh/uv/), Databricks CLI v0.205 o successivo.
- Credenziali Databricks esportate come variabili d'ambiente (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`, eventuale `DATABRICKS_CLUSTER_ID` o `DATABRICKS_SERVERLESS_COMPUTE_ID`).

## Setup locale
1. Clonare il repository e posizionarsi nella cartella radice.
2. Installare uv (se non presente):  
   ```bash
   pip install uv
   ```
3. Installare le dipendenze del progetto:  
   ```bash
   uv sync
   ```
4. Configurare le variabili d'ambiente Databricks. Esempio su PowerShell:  
   ```powershell
   setx DATABRICKS_HOST "https://<workspace>.azuredatabricks.net"
   setx DATABRICKS_TOKEN "<personal-access-token>"
   ```
   Su shell Bash usare `export`.
5. Facoltativo: creare un file `.env` per riutilizzare le credenziali durante l'esecuzione di `uv run`.
6. Validare la configurazione del bundle:  
   ```bash
   uv run databricks bundle validate --target dev
   ```

## Configurazione Databricks
- Aggiornare `src/configs/variables.py` con catalogo, schema e nome dell'endpoint LLM corretti.
- Eseguire il notebook `src/setup/prepare_environment.ipynb` per creare catalogo, schema e privilegi minimi.
- Lanciare `src/tools/create_uc_function.ipynb` per registrare le funzioni SQL che gli agenti utilizzeranno durante la validazione.
- Verificare che l'endpoint LLM indicato da `LLM_ENDPOINT_NAME` sia attivo e accessibile dal workspace.
- Se necessario, aggiornare i file YAML in `resources/` (job, pipeline) con percorsi e impostazioni specifiche dell'organizzazione.

## Esecuzione del bundle
- Deploy dell'ambiente di sviluppo:  
  ```bash
  uv run databricks bundle deploy --target dev
  ```
- Avvio del job principale (includera notebook, refresh pipeline e task wheel):  
  ```bash
  uv run databricks bundle run main_task --target dev
  ```
- Esecuzione della pipeline Delta Live Tables definita nel bundle:  
  ```bash
  uv run databricks bundle run refresh_pipeline --target dev
  ```
- Per il playground locale, lanciare il driver in `src/playground/planner_agent_with_playground/driver.py` passando eventuali variabili d'ambiente richieste.

## Test e validazione
- I test usano Databricks Connect per connettersi al workspace. Assicurarsi che il cluster remoto sia compatibile con la versione installata (`databricks-connect>=15.4,<15.5`).
- Avviare la suite:  
  ```bash
  uv run pytest
  ```
- Durante i test viene creata una sessione Spark condivisa (`tests/conftest.py`) e viene forzata la modalita serverless se non e gia definito un cluster.

## Struttura del repository
```text
.
|-- databricks.yml
|-- resources/
|   |-- synthia_data_agent.job.yml
|   `-- synthia_data_agent.pipeline.yml
|-- src/
|   |-- 1.planner_agent.ipynb
|   |-- 2.data_inspector_agent.ipynb
|   |-- agents/
|   |   `-- planner_agent.py
|   |-- configs/
|   |   |-- requirements.txt
|   |   `-- variables.py
|   |-- playground/
|   |   `-- planner_agent_with_playground/
|   |       `-- driver.py
|   |-- prompts/
|   |   |-- data_inspector.py
|   |   `-- planner.py
|   |-- setup/
|   |   `-- prepare_environment.ipynb
|   |-- tools/
|   |   `-- create_uc_function.ipynb
|   `-- typings/
|       `-- __builtins__.pyi
`-- tests/
    |-- conftest.py
    `-- main_test.py
```

## Suggerimenti operativi
- Tenere sincronizzati i prompt con le policy di sicurezza aziendali, documentando eventuali cambiamenti rilevanti.
- Registrare con MLflow le versioni stabili degli agenti e usare il Model Registry per la promozione in produzione.
- Automatizzare la creazione delle funzioni SQL tramite pipeline se devono essere mantenute su piu ambienti.
- Prima di generare dati sintetici reali, eseguire una revisione legale e di compliance per assicurarsi che i dataset di origine possano essere elaborati.

## Risorse utili
- Documentazione Databricks Asset Bundles: https://docs.databricks.com/dev-tools/bundles/
- Documentazione Databricks Connect: https://docs.databricks.com/dev-tools/databricks-connect.html
- uv package manager: https://docs.astral.sh/uv/
- MLflow Responses: https://mlflow.org/docs/latest/llms/agents/index.html

