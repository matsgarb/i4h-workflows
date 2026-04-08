# Frame Stacking Implementation for Deformable Object Manipulation

## 📋 Sommario dei Cambiamenti

Questo documento traccia l'implementazione del **frame stacking** (4 frame concatenati) per risolvere il problema di **traballamento** (instabilità) del modello di RL.

---

## 🔧 File Modificati

### 1. **observations.py**
📍 Path: `workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/liver_retraction/mdp/observations.py`

**Cosa è stato cambiato:**
- ✅ **Commentato** il codice vecchio che usava singola immagine (`camera_rgb_observation`)
- ✅ **Aggiunto** nuovo metodo: `camera_rgb_frame_stack_observation()`
- ✅ **Implementato** un buffer globale (`_FRAME_BUFFERS`) per memorizzare gli ultimi 4 frame

**Funzionamento:**
```
Input:  1 frame RGB [batch, H, W, 3]
Buffer: Memorizza gli ultimi 4 frame
Output: 4 frame concatenati [batch, H, W, 12] → ridimensionati a 84x84 → [batch, 12, 84, 84]
```

**Perché 4 frame?**
- Paper: "Sim-To-Real Transfer for Visual RL of Deformable Object Manipulation"
- Il tessuto deformabile è parzialmente osservabile (POMDP)
- 4 frame permettono alla rete di inferire: **velocità + direzione del movimento**
- Con 1 frame solo: il robot non sa se il tessuto si sta muovendo → micro-correzioni avanti/indietro (traballamento)

---

### 2. **reach_env_cfg.py**
📍 Path: `workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/liver_retraction/reach_env_cfg.py`

**Cosa è stato cambiato:**
- ✅ **Commentato**: `func=mdp.camera_rgb_observation` (singola immagine)
- ✅ **Attivato**: `func=mdp.camera_rgb_frame_stack_observation` (frame stacking)

**Osservazione della policy:**
```python
@configclass
class PolicyCfg(ObsGroup):
    camera_rgbd = ObsTerm(func=mdp.camera_rgb_frame_stack_observation)
    # Output: [batch, 12, 84, 84]
```

---

### 3. **rsl_rl_cfg.py**
📍 Path: `workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/liver_retraction/config/psm/agents/rsl_rl_cfg.py`

**Cosa è stato cambiato:**
- ✅ **Aggiunto parametro**: `input_channels=12` alla configurazione della CNN
- ✅ **Documentato**: Commenti che spiegano il perché (4 frame * 3 canali RGB = 12)

**CNN Configuration:**
```python
policy = RslRlPpoActorCriticCNNCfg(
    class_name="ActorCriticCNN",
    input_channels=12,  # ← IMPORTANTE: 4 frame * 3 RGB = 12 canali
    actor_cnn_cfg={
        "output_channels": [32, 64, 64],
        "kernel_size": [8, 4, 3],
        "stride": [4, 2, 1],
        "activation": "elu",
    },
    ...
)
```

---

## 📊 Input/Output Shapes

### Frame Stacking Pipeline:
```
Camera Output:
  ├─ Frame t-3: [batch, H, W, 3]
  ├─ Frame t-2: [batch, H, W, 3]
  ├─ Frame t-1: [batch, H, W, 3]
  └─ Frame t:   [batch, H, W, 3]
  
Buffer Concatenation:
  └─→ [batch, H, W, 12] (H,W sono ~320x240 dal sensore)

Resize (interpolate):
  └─→ [batch, 12, 84, 84] (NCHW format after permute)

Policy Input:
  └─→ ActorCriticCNN accepts [batch, 12, 84, 84]
```

---

## 🧪 Come Testare l'Implementazione

### 1. **Verifica che il frame stacking funziona:**
```bash
cd /home/dvrkteam/i4h-workflows/workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl

# Training con frame stacking
python train.py --task Isaac-Liver-PSM-Reach-v0 --num_envs 4 --enable_cameras
```

### 2. **Monitor degli output:**
Nei log dovresti vedere:
- Shape dell'osservazione: `[batch, 12, 84, 84]` ✅
- Nessun errore di shape mismatch nella CNN

### 3. **Backward Compatibility:**
- La vecchia funzione `camera_rgb_observation()` è ancora disponibile (commentata)
- Puoi switchar tra le due modificando `reach_env_cfg.py` se necessario

---

## 🎯 Risultati Attesi

### Prima (singola immagine):
```
❌ Instabilità: il robot oscilla avanti e indietro
❌ Traballamento visibile nel movimento della pinza
❌ Difficoltà a mantenere il movimento fluido
```

### Dopo (frame stacking):
```
✅ Movimento fluido e stabile
✅ La rete inferisce la velocità del tessuto
✅ Azioni coerenti e non tremolanti
✅ Migliore stabilità nel raggiungere il target
```

---

## 📚 Riferimenti

**Paper:** "Sim-To-Real Transfer for Visual Reinforcement Learning of Deformable Object Manipulation"

**Sezione chiave:** II.B.3 - Observations
- "The four most recent images are concatenated"
- Input: 84×84×12 (4 RGB images)

**Ispirato da:** DQN (Mnih et al., Nature 2015)
- Frame stacking per capire la dinamica in POMDP

---

## ⚠️ Note Importanti

1. **Prima del training:** Assicurati che `input_channels=12` sia impostato nella CNN config
2. **Buffer globale:** I frame buffer vengono mantenuti in memoria durante l'episodio
3. **Reset:** Il buffer si riempie ripetendo il primo frame al reset dell'ambiente
4. **Performance:** Nessun overhead significativo (solo concatenazione tensori)

---

## 🔄 Come Tornare alla Versione Vecchia (se necessario)

In `reach_env_cfg.py`, scambia:
```python
# Commentate la nuova versione
# camera_rgbd = ObsTerm(func=mdp.camera_rgb_frame_stack_observation)

# Abilita la vecchia versione
camera_rgbd = ObsTerm(func=mdp.camera_rgb_observation)
```

E in `rsl_rl_cfg.py`:
```python
# Rimuovi o commenta:
# input_channels=12,

# Oppure imposta:
input_channels=3  # Torna a singola immagine
```

---

**Data:** 4 Marzo 2026  
**Implementazione:** Frame Stacking per Visual RL  
**Status:** ✅ Pronto per il training
