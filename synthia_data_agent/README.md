# Synthia Data Agent
## Introduzione
Synthia Data Agent è un agente intelligente in ambiente Databricks progettato per generare dati sintetici realistici in modo interattivo e sicuro.

L’agente interpreta il linguaggio naturale dell’utente, identifica il tipo di dataset da creare e utilizza modelli generativi (come CTGAN, CopulaGAN o TVAE) per produrre dati che imitano la struttura e le distribuzioni dei dati reali, preservando la privacy.

I dataset generati vengono validati, salvati in Delta Lake e resi disponibili per analisi, test o addestramento di modelli di machine learning.

🧬 Dati sintetici\
Sono dati artificiali generati da algoritmi o modelli di intelligenza artificiale, progettati per imitare i pattern statistici dei dati reali senza contenere informazioni sensibili o identificabili.

🧠Esempio\
Se hai un dataset reale — le vendite di un negozio — un sistema di generazione sintetica può creare un nuovo dataset con gli stessi pattern, distribuzioni e correlazioni, ma senza contenere nessun dato reale di clienti o transazioni.

A cosa servono?
1. Privacy e conformità
2. Data augmentation
3. Test e sviluppo
4. Simulazione e ricerca


