# MMAI3 – Montagsmaler mit KI (CNN + CLIP, erweiterte Klassen)

MMAI3 ist die dritte Version des Projekts *Montagsmaler mit KI* und baut direkt auf MMAI2 auf.  
Wie in der vorherigen Version werden sowohl ein **selbst trainiertes CNN-Modell** als auch ein **CLIP-Modell** eingesetzt.

Der Fokus von MMAI3 liegt auf der **Erweiterung der verwendeten Klassen und Bildbeschreibungen** sowie auf der Untersuchung, wie sich eine größere Anzahl an möglichen Begriffen auf die Stabilität und Genauigkeit der Vorhersagen auswirkt.

---

## Features

- Zeichnen im Browser (HTML5 Canvas)
- KI-Vorhersagen durch zwei Modelle:
  - CNN (trainierte Klassen)
  - CLIP (Text–Bild-Vergleich, Zero-Shot)
- Anzeige von Konfidenzwerten
- Top-1- und Top-3-Vorhersagen
- Erweiterte CLIP-Bildbeschreibungen (größeres Vokabular)
- Vergleich der Ergebnisse von CNN und CLIP
- Speicherung der Zeichnungen inkl. Modellvorhersagen
- Galerie mit gespeicherten Ergebnissen

---

## Verwendete Technologien

- **TensorFlow / Keras** – Training des CNN
- **FastAPI** – Backend und Modell-Inferenz
- **Python** – Datenverarbeitung, Training und Backend-Logik
- **HTML / CSS / JavaScript** – Frontend
- **Google Quick, Draw! Dataset** – Trainingsdaten für das CNN
- **OpenAI CLIP** – Zero-Shot Image–Text Matching
- **PyTorch** – Ausführung des CLIP-Modells

---

## Quick Draw! NDJSON-Dateien hinzufügen

Die **NDJSON-Dateien des Quick Draw!-Datensatzes müssen manuell heruntergeladen und eingefügt werden**, da sie aus Größengründen nicht im Repository enthalten sind.

Lade die gewünschten Kategorien von:  
https://github.com/googlecreativelab/quickdraw-dataset

Die verwendeten Kategorien müssen mit den in `class_indices.json` definierten Klassen übereinstimmen.  
Zusätzlich werden für CLIP erweiterte Bildbeschreibungen aus einer `captions.txt`-Datei genutzt.


## How to get started

1. Virtuelle Umgebung erstellen und aktivieren
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate    # macOS / Linux
```
2. Abhängigkeiten installieren
```bash
    pip install -r requirements.txt
```
3. NDJSON → Bilder konvertieren
```bash
    cd backend
    python convert_ndjson_to_png.py
```
4. KI trainieren
```bash
    python train_model.py
```
5. Backend starten (FastAPI)
```bash
    uvicorn backend.main:app --reload --port 8003
```
Test (optional):
    Browser öffnen:
    http://127.0.0.1:8003/docs

6. Frontend starten
```bash
    frontend/index.html
```

