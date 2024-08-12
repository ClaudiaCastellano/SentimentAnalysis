# SentimentAnalysis

## Descrizione
SentimentAnalysis è un progetto realizzato per l'esame di Cognitive Computing Systems. 
Il progetto consiste nello sviluppo di un modello predittivo in grado di estrarre e analizzare i sentimenti espressi nei testi utilizzando tecniche di Natural Language Processing (NLP) e deep learning. 

Il modello è stato addestrato sulla base del dataset 'Sentiment140' che contiene 1.600.000 tweets etichettati come negativi (sentiment = 0) e positivi (sentiment = 4).

Il file [SentimentAnalysis.ipynb](./SentimentAnalysis.ipynb) contiene il codice utilizzato per:
- import delle librerie necessarie
- caricamento e analisi del dataset
- text-preprocesing
- realizzazione delle word cloud
- suddivisione del dataset in train, validation e test
- tokenization, label encoding e word embedding
- costruzione e addestramento della rete neurale 
- plot delle curve di addestramento
- visualizzazione della matrice di confusione

Il file [app.py](./app.py) è uno script Python che, utilizzando la libreria Gradio, avvia un server locale sulla porta 7872 in cui viene ospitata un'interfaccia web che pemette agli utenti di testare in maniera interattiva il modello. Essi infatti, inserendo una frase possono visualizzare il sentiment (positivo o negativo) che il modello associa ad essa.







