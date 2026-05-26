A good way to frame the project is not immediately as “train a classifier”, but as:



  ▌ Can acoustic features from individual chicken vocalizations explain White Striping severity beyond body weight, carcass weight, and diet?                                                                                   


  Given your dataset size and class imbalance, this should be treated primarily as an exploratory/statistical + small-ML study, not a deep-learning prediction project.


  1. Define the realistic targets


  White Striping: usable but imbalanced


  You have scores approximately:



   • 0: no problem
   • 0.5: mild/prevalent
   • 1: moderate
   • 2: severe, but only one case


  For White Striping you can try:



   1 Ordinal outcome
     Use the score as ordered severity: 0 < 0.5 < 1 < 2.



   2 Binary outcome
     For example:
      • WS = 0 vs WS > 0
      • or WS < 1 vs WS >= 1

 # A good way to frame the project is not immediately as “train a classifier”, but as:



  ▌ Can acoustic features from individual chicken vocalizations explain White Striping severity beyond body weight, carcass weight, and diet?                                                                                    



  Given your dataset size and class imbalance, this should be treated primarily as an exploratory/statistical + small-ML study, not a deep-learning prediction project.



 ## 1. Define the realistic targets



 ### White Striping: usable but imbalanced



 You have scores approximately:




  • 0: no problem
  • 0.5: mild/prevalent
  • 1: moderate
  • 2: severe, but only one case



 For White Striping you can try:




 1 Ordinal outcome
   Use the score as ordered severity: 0 < 0.5 < 1 < 2.




 2 Binary outcome
   For example:
    • WS = 0 vs WS > 0
    • or WS < 1 vs WS >= 1



 The best choice depends on how many animals fall into each group.



 ### Wooden Breast and Spaghetti Meat: not suitable for prediction



 If only 2 chickens are positive for each, you cannot build a reliable model for those outcomes. You can still do:




  • descriptive comparison,
  • case-study analysis,
  • check whether those two animals have extreme audio features,



 but not supervised classification.



 ## 2. Build one row per chicken



 Your final modelling table should have exactly one row per chicken:




                                                                                                  
  Chicken ID  Diet  Live weight  Carcass weight  WS score  WB score  SM score  audio features... 
  ────────────────────────────────────────────────────────────────────────────────────────────── 
                                                                                                   



 The .wav files are about 2 minutes long, so you should not treat every audio window as an independent chicken. If you split one recording into many chunks, you must aggregate those chunks back to the chicken level.



 Otherwise you would artificially inflate the sample size.



 ## 3. Extract interpretable audio features first



 Before using deep learning, extract classical acoustic features. They are more appropriate for 296 samples and easier to interpret biologically.



 For each .wav file:



 ### Basic features




  • duration,
  • mean volume/RMS energy,
  • silence percentage,
  • number of high-energy vocal events,
  • average vocal-event duration,
  • vocalization rate per minute.



 ### Spectral features




  • spectral centroid,
  • spectral bandwidth,
  • spectral rolloff,
  • spectral flatness,
  • zero-crossing rate.



 ### MFCC features



 Extract 20 MFCCs over short windows, then summarize each MFCC by:




  • mean,
  • standard deviation,
  • minimum,
  • maximum,
  • median,
  • 10th percentile,
  • 90th percentile.



 This gives a compact representation of the overall vocal profile of each chicken.



 ### Possible pitch-related features



 If the recordings are clean enough:




  • estimated fundamental frequency,
  • mean pitch,
  • pitch variation,
  • maximum pitch,
  • minimum pitch.



 However, chicken vocalizations can be noisy and irregular, so pitch extraction may not always be reliable.



 ## 4. Segment the audio, then aggregate



ML intern plan — ottimizzato

Obiettivo: verificare se caratteristiche acustiche ricavate dalle vocalizzazioni individuali spiegano la severità di White Striping (WS) oltre a dieta, peso vivo e peso di carcassa. Trattare lo studio come esplorativo/statistico + piccoli modelli ML, non deep learning.

## 1. Target realistici
- White Striping: utilizzabile ma sbilanciato (es. 0, 0.5, 1, 2 — 2 molto raro).
- Possibili formulazioni:
  - Ordinale: 0 < 0.5 < 1 < 2
  - Binario: es. WS = 0 vs WS > 0, oppure WS < 1 vs WS >= 1

Nota: Wooden Breast (WB) e Spaghetti Meat (SM) hanno pochissimi casi (es. 2) — non adatti a modelli supervisati, solo analisi descrittive/case study.

## 2. Tabella a livello di animale
Costruire una riga per ogni pollo: ID, dieta, peso vivo, peso carcassa, punteggi (WS, WB, SM), feature audio aggregate. Non trattare frammenti audio come osservazioni indipendenti: aggregare per pollo.

## 3. Estrazione feature acustiche (per .wav)
- Basic: durata, RMS energy medio, frazione di silenzio, numero eventi ad alta energia, durata media eventi, tasso vocalizzazioni/min.
- Spettrali: spectral centroid, bandwidth, rolloff, flatness, zero-crossing rate.
- MFCC: estrarre 20 MFCC su finestre brevi; aggregare ogni MFCC con mean, std, min, max, median, p10, p90.
- Pitch (se affidabile): f0 stimata, mean pitch, var pitch, max, min.

## 4. Segmentazione e aggregazione
Procedura pratica: converti in mono, resample 16 kHz o 22.05 kHz, split in finestre 2–5 s, rimuovi finestre estremamente silenziose (ma tieni la frazione di silenzio come feature), estrai feature per finestra e aggrega (mean, std, percentili) per pollo.

## 5. Analisi statistica iniziale
- Calibri: correlazione di Spearman tra feature audio e punteggio WS (tratta come ordinale).
- Controlli: eseguire correlazioni parziali o regressioni controllando per peso vivo, peso carcassa e dieta per verificare se le audio-feature aggiungono informazione oltre i metadata.

## 6. Confronto modelli
Confrontare tre insiemi di feature:
- Modello A (metadata): dieta, peso vivo, peso carcassa (baseline)
- Modello B (audio): MFCC, feature spettrali, tasso vocalizzazioni, frazione silenzio
- Modello C (metadata + audio): unione A+B — testare se C migliora A

Modelli raccomandati (dato N≈296): logistic regression (L1/L2/elastic-net), SVM, random forest/gradient boosting come secondari. Per ordinale: ordinal logistic, ridge, RF regression, GB regression.

Interpretabilità: coefficienti regressione, permutation importance, SHAP con cautela, matrici di correlazione.

## 7. Validazione
- Use: cross-validation a livello di pollo (repeated stratified 5-fold consigliato). Se usi frammenti, assicurati che tutti i frammenti dello stesso pollo siano nello stesso fold.
- Metriche binarie: balanced accuracy, ROC-AUC, precision/recall, F1, confusion matrix.
- Metriche ordinale/regressione: Spearman tra predetto/osservato, MAE, quadratic weighted kappa (se categoriale ordinato).

## 8. Distribuzione dei punteggi
Considerare due analisi principali: presenza/assenza (WS=0 vs WS>0) e distinzione leve più gravi (WS<1 vs WS>=1), a seconda delle conte per classe.

## 9. WB e SM — analisi descrittiva
Per WB/SM con pochissimi casi: identificare i polli positivi, plottare le loro feature rispetto alla distribuzione della popolazione e riportare risultati descrittivi (no inferenza statistica robusta).

## 10. Workflow suggerito
1. Data audit: missing, ID duplicati, match .wav ↔ Excel, distribuzioni punteggi e dieta, pesos.
2. Exploratory plots: WS vs peso vivo/carcassa, WS per dieta, WS vs feature audio principali.
3. Estrazione feature audio → `audio_features.csv` (una riga per pollo).
4. Merge metadata + audio → `final_dataset.csv`.
5. Analisi statistica (Spearman, parziale, regressioni controllate).
6. Modelli predittivi (A vs B vs C) con CV.
7. Interpretazione e report: miglioramento apportato dalle audio-feature oltre i metadata.

## 11. Analisi avanzata opzionale
After the classical feature analysis, si possono provare embedding preaddestrati (wav2vec2, HuBERT, WavLM, VGGish, PANNs): estrarre embedding da finestre, aggregare per pollo e usare modelli regolarizzati senza fine-tuning.

## 12. Ipotesi principale
"Investighiamo se feature acustiche derivate dalle vocalizzazioni individuali sono associate alla severità di White Striping e se migliorano la spiegazione/predizione oltre dieta, peso vivo e peso di carcassa."

## 13. Output minimo raccomandato
1. Dataset pulito.
2. Statistiche riassuntive per i punteggi.
3. Tabella feature audio.
4. Matrice di correlazione audio vs WS.
5. Modelli: metadata-only, audio-only, metadata+audio.
6. Performance cross-validated.
7. Plot importanza feature.
8. Analisi descrittiva per WB e SM.

---
Versione compatta e senza newline/spazi ridondanti.