# progetto_mobd

## Librerie
```
pip install numpy
pip install scikit-learn
```

## Training
Il file dei dati di training `training_set.csv` deve essere nella stessa directory di `train.py`.
Per il training occorre eseguire il file `train.py` con il comando 
```
python train.py
```
che salverà nella stessa directory i file `preprocess.pkl` e `model.pkl` contenenti rispettivamente il preprocessing ed il modello addestrato. 

## Test
Occorre mettere il file `test_set.csv` nella stessa directory di `test.py` ed eseguire quest'ultimo script con il comando 
```
python test.py
```
Questo stamperà a schermo il classification report.
