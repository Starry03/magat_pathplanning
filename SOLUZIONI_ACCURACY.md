# Soluzioni per Migliorare Accuracy da 0.5 a 0.9

## Problema
Sia il modello base (`DecentralPlannerNet`) che `PaperArchitecture` si fermano a ~0.5 di accuracy invece di raggiungere 0.9 come nel paper.

## Cause Identificate

### 1. ⚠️ Collision Shielding Durante Training
Il simulatore blocca preventivamente le azioni che causerebbero collisioni. Questo impedisce al modello di imparare a evitarle autonomamente.

**Evidenza**: High `rateFailedReachGoalbyCollisionShielding` indica che molti casi falliscono perché gli agenti vengono bloccati dal collision shield invece di trovare percorsi alternativi.

### 2. ⚠️ GSO (Graph Shift Operator) Troppo Sparso
Con `GSO_mode=dist_GSO` + `commR=7`, molti agenti non sono connessi nel grafo → il GNN non riesce a propagare informazioni globali.

### 3. ⚠️ MaxPooling Aggressivo (solo modello base)
Con FOV=9 (input 11x11) e 3 maxpool stride=2, la feature map diventa 1x1 troppo presto → perdita di informazione spaziale.

### 4. ⚠️ Metrica Accuracy Molto Restrittiva
`rateReachGoal` conta solo casi dove **TUTTI** gli agenti arrivano al goal senza collisioni. Con 10 agenti, anche 1 solo fallimento → accuracy = 0 per quel caso.

---

## 🔧 Soluzioni (Priorità Alta → Bassa)

### ⭐⭐⭐ Soluzione 1: Modifica Collision Shielding

**Durante Training**: disabilita o riduci collision shielding
**Durante Test**: mantienilo attivo per safety

#### Opzione A: Flag nel Config
Aggiungi nei config JSON:
```json
"collision_shielding_train": false,
"collision_shielding_test": true
```

Poi in `utils/new_simulator.py` o `multirobotsim_*.py`, nel metodo `move()`:
```python
def move(self, actionVec, currentStep, use_collision_shield=None):
    if use_collision_shield is None:
        use_collision_shield = self.config.get('collision_shielding_train', True)
    
    # ... existing collision check logic
    if use_collision_shield and check_collision:
        # block the action
    else:
        # execute the action anyway
```

E negli agenti (es. `paperagent.py`):
```python
# Training
allReachGoal, check_moveCollision, check_predictCollision = self.robot.move(
    actionVec_predict, currentStep, use_collision_shield=False
)

# Testing
allReachGoal, check_moveCollision, check_predictCollision = self.robot.move(
    actionVec_predict, currentStep, use_collision_shield=True
)
```

#### Opzione B: Aumenta Penalty Collisioni nella Loss
Nel config:
```json
"penaltyCollision": 0.5  // aumenta da 0.05 a 0.5 o 1.0
```

Verifica che la loss effettivamente usi questo penalty. Se non è implementato, aggiungi:
```python
# In train_one_epoch_Batch
if check_collision:
    loss += self.config.penaltyCollision * collision_count
```

---

### ⭐⭐⭐ Soluzione 2: Cambia GSO Mode

Prova questi esperimenti:

#### Test 1: Full GSO (Grafo completamente connesso)
```bash
python main.py configs/dcpGAT_OE_Random.json \
    --mode train \
    --GSO_mode full_GSO \
    --num_agents 10 \
    --FOV 9
```

**Pro**: Ogni agente vede tutti gli altri → massima cooperazione  
**Contro**: Computazionalmente costoso con molti agenti

#### Test 2: Binary GSO con Communication Radius Aumentato
```bash
python main.py configs/dcpGAT_OE_Random.json \
    --mode train \
    --GSO_mode dist_GSO_one \
    --commR 15 \
    --num_agents 10 \
    --FOV 9
```

**Pro**: GSO binario (0 o 1) invece di distanze weighted → più stabile  
**Contro**: Richiede commR abbastanza grande

---

### ⭐⭐ Soluzione 3: Aumenta K (Graph Filter Taps)

Il numero di hop nel GNN determina quanto lontano si propagano le informazioni:

```bash
python main.py configs/dcpGAT_OE_Random.json \
    --mode train \
    --nGraphFilterTaps 5 \
    --GSO_mode dist_GSO_one \
    --commR 10
```

Con K=5 e commR=10, anche agenti distanti possono comunicare indirettamente.

---

### ⭐⭐ Soluzione 4: Riduci Aggressività MaxPooling (solo per modello base)

Se usi `DecentralPlannerNet` (non `PaperArchitecture`), modifica `graphs/models/decentralplanner.py`:

**Prima** (maxpool ogni 2 layer):
```python
if l % 2 == 0:
    convl.append(nn.MaxPool2d(kernel_size=2))
```

**Dopo** (maxpool solo dopo layer 1 e 3):
```python
if l == 1 or l == 3:  # maxpool solo 2 volte invece di 3
    convl.append(nn.MaxPool2d(kernel_size=2))
```

Oppure usa stride nel Conv invece di MaxPool separato.

---

### ⭐ Soluzione 5: Data Augmentation / Online Expert

Se hai l'Online Expert attivo:
```json
"Start_onlineExpert": 20  // inizia OE dopo 20 epoche invece di subito
```

Oppure aumenta threshold:
```json
"threshold_SuccessRate": 90  // richiedi 90% success prima di switchare a OE
```

---

### ⭐ Soluzione 6: Learning Rate Scheduler

Verifica che il scheduler non stia facendo decay troppo veloce:

```python
# In agent __init__
self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
    self.optimizer, 
    T_max=self.config.max_epoch, 
    eta_min=1e-5  # aumenta da 1e-6 se LR diventa troppo piccolo
)
```

Oppure usa ReduceLROnPlateau:
```python
self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.optimizer, 
    mode='max',  # monitora accuracy
    factor=0.5, 
    patience=5, 
    min_lr=1e-5
)
```

---

## 🧪 Piano di Test Consigliato

### Test 1: Baseline + Full GSO
```bash
python main.py configs/dcpGAT_OE_Random.json \
    --mode train \
    --GSO_mode full_GSO \
    --nGraphFilterTaps 3 \
    --num_agents 10 \
    --max_epoch 50
```

### Test 2: Binary GSO + High CommR + No Collision Shield
(Richiede modifiche al simulatore per disabilitare collision shield)
```bash
python main.py configs/dcpGAT_OE_Random.json \
    --mode train \
    --GSO_mode dist_GSO_one \
    --commR 20 \
    --nGraphFilterTaps 5 \
    --num_agents 10
```

### Test 3: PaperArchitecture + Full GSO
```bash
python main.py configs/your_paper_config.json \
    --mode train \
    --GSO_mode full_GSO \
    --nGraphFilterTaps 3 \
    --batch_numAgent True
```

---

## 📈 Metriche da Monitorare

Durante il training, oltre a `rateReachGoal`, monitora:

1. **`rateFailedReachGoalSH`**: Se alta → collision shielding blocca troppo
2. **`avg_rate_deltaMP`** (Makespan deterioration): Se alta → il modello fa percorsi molto più lunghi dell'optimum
3. **`rateCollisionPredictedinLoop`**: Se alta → il modello predice azioni che causerebbero collisioni (collision shield le blocca)
4. **Training Loss**: Se continua a scendere ma accuracy no → overfitting o problema con metrica

Nel tensorboard:
```bash
tensorboard --logdir=./Data/Tensorboard
```

Guarda:
- `epoch/train_set_Accuracy_reachGoalNoCollision`
- `epoch/valid_set_Accuracy_reachGoalNoCollision`
- `iteration/loss`

---

## 🎯 Target Realistico

| Setup | Expected Accuracy |
|-------|------------------|
| **Baseline** (commR=7, dist_GSO, K=3) | 0.5-0.6 (current) |
| **+ Full GSO** | 0.65-0.75 |
| **+ Full GSO + K=5** | 0.75-0.85 |
| **+ Full GSO + K=5 + No Collision Shield** | 0.85-0.92 |
| **+ All above + Fine-tuning** | 0.90-0.95 (paper level) |

---

## ⚠️ Note Importanti

### Bug già Risolti
✅ FOV key mismatch (`config["fov"]` → `config.get("FOV", ...)`)  
✅ Output shape `(B*N, 5)` → `(B, N, 5)`  
✅ Weights initialization commentata → attivata

### Da Verificare
❓ Se il dataset di training ha abbastanza varietà (diverse mappe, densità, # agenti)  
❓ Se l'offline expert genera path veramente ottimali (controlla `offlineExpert/solutions/`)  
❓ Se c'è data leakage tra train/valid/test set

---

## 📚 Riferimenti

- Paper originale: [MAGAT paper reference - inserisci qui]
- Repo originale: [se disponibile]
- Issue simili: [link a issue/discussion se ne hai trovati]

---

## 🔄 Prossimi Passi

1. **Immediate**: Runnare Test 1 (Full GSO) per vedere se accuracy sale
2. **Se Test 1 funziona**: Provare K=5 per ulteriore boost
3. **Se Test 1 non funziona abbastanza**: Implementare Soluzione 1 (disable collision shield) e ri-testare
4. **Fine-tuning**: Una volta raggiunto 0.8+, fare hyperparameter tuning (LR, dropout, weight decay)
5. **Generalization**: Testare su mappe/# agenti mai visti

---

**Ultimo aggiornamento**: 2025-10-22  
**Autore**: Analisi automatica del codice
