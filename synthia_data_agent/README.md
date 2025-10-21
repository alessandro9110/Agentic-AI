
# Synthia Data Agent

## Panoramica
Synthia Data Agent è un piccolo ecosistema di agenti progettato per aiutare a pianificare, validare e preparare la generazione di dataset sintetici su Databricks. L'obiettivo principale è trasformare richieste in linguaggio naturale in specifiche strutturate di dataset, verificare le risorse sorgente in Unity Catalog e produrre metadata utili per i successivi passi di generazione (modelli GAN/TVAE, pipeline DLT, ecc.).

L'agente non esegue direttamente la generazione sintetica dei dati: si occupa della fase di analisi e orchestrazione (planning, validazione risorse, raccolta metadati). Questo rende il componente riutilizzabile in pipeline più ampie dove il passo di generazione può essere delegato a moduli separati.

## Cosa fa questo agente (in breve)

- Riceve input in linguaggio naturale (es. "Genera un dataset sintetico di clienti con colonne X, Y, Z")
- Pianifica la struttura target del dataset (tipi, cardinalità, colonne sensibili, trasformazioni richieste)
- Verifica l'esistenza e lo stato delle risorse in Unity Catalog (cataloghi, schemi, tabelle)
- Raccoglie metadati della tabella sorgente (schema, tipi di dato, statistiche di base)
- Produce una specifica strutturata che può essere passata a moduli di generazione o a pipeline Databricks
- Registra/tracka output e decisioni (possibile integrazione con MLflow Responses o simili)

## Sub-agenti

Questa sezione descrive i sub-agenti che compongono il Synthia Data Agent. I sub-agenti sono componenti specializzati invocati dall'orchestrator/coordinatore principale (es. il Planner orchestratòr o il framework che esegue il bundle Databricks). Ognuno ha un ruolo circoscritto (analisi, validazione, raccolta metadati) e definisce un piccolo "contratto" input/output per facilitare l'integrazione.

### Planner (sub-agent)

- Scopo: ricevere una richiesta in linguaggio naturale e trasformarla in una specifica strutturata di dataset pronta per la generazione o per essere passata a un orchestrator. Definisce colonne target, tipi, vincoli di privacy/sensibilità, dimensionamento stimato e trasformazioni richieste.
- Dove si trova: implementazione principale in `src/agents/planner_agent.py` (notebook dimostrativo: `src/1.planner_agent.ipynb`).
- Come viene invocato: tipicamente chiamato dal coordinatore principale quando arriva una richiesta utente o come parte di una pipeline di orchestrazione.
- Contratto (sintetico):
  - Input: testo libero (richiesta utente) + opzionali parametri di contesto (catalogo/schema/tabella, vincoli di privacy).
  - Output: oggetto strutturato (es. JSON) con campi come `columns`, `types`, `sensitive_columns`, `estimated_rows`, `transformations`, `notes`.
  - Modalità di errore: input ambiguo, mancanza di contesto o incongruenze tra richieste e risorse disponibili. Deve restituire messaggi diagnostici che l'orchestrator può loggare o mostrare all'utente.

### Data Inspector (sub-agent)

- Scopo: ispezionare e validare risorse sorgente in Unity Catalog, raccogliendo metadati e statistiche utili (schema, tipi, distribuzioni sommarie, null ratio, candidate key). Queste informazioni aiutano il Planner a produrre specifiche realistiche.
- Dove si trova: definito come flusso di ispezione nei notebook e prompt (`src/2.data_inspector_agent.ipynb`, `src/prompts/data_inspector.py`). Alcune funzionalità usano utility in `src/tools/` per registrare e chiamare funzioni SQL su Databricks.
- Come viene invocato: chiamato dal planner o dall'orchestrator quando è necessario validare una risorsa sorgente o raccogliere statistiche.
- Contratto (sintetico):
  - Input: identificatore risorsa (catalog, schema, table) e credenziali/contesto per interrogare Unity Catalog/SQL.
  - Output: metadati strutturati (schema colonne, tipi, valori mancanti per colonna, statistiche di base come min/max/avg, distribuzioni approssimate, suggerimenti su colonne sensibili).
  - Modalità di errore: permessi insufficienti, risorsa non trovata, query troppo pesante o timeout. In questi casi il sub-agent ritorna messaggi diagnostici e suggerisce azioni (es. eseguire statistiche campionate o richiedere permessi).


## Contenuto della cartella `src/`

Di seguito una panoramica delle sottocartelle e dei file principali all'interno di `src/` con una breve spiegazione del loro scopo.

- `src/agents/`
  - Contiene l'implementazione degli agenti programmati. Al momento principale è il `planner_agent.py` che riceve i messaggi utente, applica i prompt corretti e costruisce la richiesta strutturata (usando LangGraph/LangChain come orchestratori).

- `src/configs/`
  - File di configurazione e variabili condivise per l'agente. Per esempio `variables.py` contiene nomi di catalogo/schema/endpoint LLM da aggiornare per l'ambiente, mentre `requirements.txt` raccoglie dipendenze specifiche per il sotto-modulo.

- `src/playground/`
  - Esempi, driver e script di test/esperimenti. In particolare questa cartella include esperimenti e test realizzati utilizzando l'AI Playground di Databricks (notebook/driver che riproducono scenari interattivi, chiamate all'endpoint LLM, e flow di debug). Utile per riprodurre demo rapide o per sviluppo iterativo prima del deploy su Databricks.

- `src/prompts/`
  - Prompt testuali usati dagli agenti (planner, data inspector). I prompt sono il punto di integrazione tra il codice e il modello LLM e vanno mantenuti sincronizzati con le policy aziendali.

- `src/setup/`
  - Notebook e script di setup che aiutano a preparare l'ambiente Databricks (creazione cataloghi, privilege grants, ecc.). Eseguire questi notebook prima di lanciare i job/pipeline in ambienti nuovi.

- `src/tools/`
  - Utility e notebook che registrano funzioni SQL di supporto (per esempio funzioni che verificano l'esistenza di cataloghi, schemi e tabelle in Unity Catalog). Queste funzioni vengono poi invocate dagli agenti durante la validazione.

- `src/typings/`
  - Tipizzazioni e file stub utili durante lo sviluppo per migliorare l'editor experience e l'analisi statica.

- Notebook principali
  - `src/1.planner_agent.ipynb`: notebook dimostrativo che mostra come usare il planner agent passo-passo.
  - `src/2.data_inspector_agent.ipynb`: notebook per la validazione delle risorse tramite i prompt del data inspector.

## Esempio di file chiave

- `src/agents/planner_agent.py`: implementa la logica di parsing del prompt e produzione della specifica di dataset.
- `src/configs/variables.py`: punto unico per configurare catalogo, schema e nome dell'endpoint LLM.
- `src/tools/create_uc_function.ipynb`: notebook che registra le funzioni SQL (es. `check_catalog_exist`, `check_schema_exist`, `check_table_exist`) nello schema dedicato.

## Come utilizzare al volo (nota rapida)

- Aggiornare `src/configs/variables.py` con i valori del workspace Databricks e l'endpoint LLM.
- Eseguire i notebook in `src/setup/` per predisporre cataloghi e privilegi.
- Usare i notebook `src/1.planner_agent.ipynb` e `src/2.data_inspector_agent.ipynb` per testare il flusso locale o eseguire i driver in `src/playground/`.

## Esecuzione dei test

- La suite di test (`tests/`) usa Databricks Connect per connettersi al workspace remoto nelle integrazioni. Seguire le istruzioni nel README principale per configurare `DATABRICKS_HOST` e `DATABRICKS_TOKEN` prima di eseguire `uv run pytest`.

## Note operative e suggerimenti

- Tenere i prompt aggiornati e verificati contro le policy aziendali.
- Tracciare le decisioni chiave con MLflow o simili quando si passa in produzione.
- Automatizzare la registrazione delle funzioni SQL e la creazione del catalogo tramite i bundle Databricks quando si promuove in ambienti diversi.

---

Se vuoi, posso:
-- Aggiungere esempi concreti di input/OUTPUT del `planner_agent` nel README.
-- Creare una checklist di setup automatica (script/Makefile) per eseguire i notebook di setup e registrare le funzioni SQL.

