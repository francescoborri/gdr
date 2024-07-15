# Progetto Gestione di Rete

## Prerequisiti

Prima di eseguire il progetto è necessario creare l'ambiente virtuale ed installare le dipendenze.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Istruzioni per l'esecuzione

Gli *script* eseguibili sono quattro e in ognuno di essi è possibile visualizzare le opzioni disponibili tramite il comando `python3 <script>.py --help`.

- `exp_smoothing.py`: realizza la previsione utilizzando il modello *Holt-Winters*.
- `analyze.py`: permette di visualizzare e analizzare una serie temporale memorizzata in un database RRD al fine di determinare la stagionalità e i parametri migliori per il modello ARIMA.
- `arima.py`: realizza la previsione utilizzando il modello ARIMA.
- `auto_arima.py`: realizza la previsione utilizzando il modello ARIMA (o SARIMA) con _fitting_ automatico dei parametri.
