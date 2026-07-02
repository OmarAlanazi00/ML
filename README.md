# Student Depression Prediction — Streamlit App

This app mirrors the full notebook workflow (EDA → cleaning → preprocessing →
PCA → modeling → evaluation) and adds a live prediction tool.

## 1. Files
- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- **You need to add** `student_lifestyle_100k.csv` (the dataset used in the notebook)
  in the same folder as `app.py`. If it's not found, the app will show a file
  uploader so you can upload it manually at runtime.

## 2. Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## 3. Deploy for free (Streamlit Community Cloud)
1. Push `app.py`, `requirements.txt`, and `student_lifestyle_100k.csv` to a
   public GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, set the main file to `app.py`.
4. Click **Deploy** — you'll get a public URL to submit.

## 4. What's inside the app (tabs)
1. **Data Overview** — dataset description + sample rows + summary stats.
2. **EDA** — distribution plots, boxplots by Depression, correlation heatmap.
3. **Cleaning & Preprocessing** — missing/duplicate checks, feature
   engineering (`sleep_category`, `study_balance`), encoding, scaling
   explanation, PCA variance plot.
4. **Model Results** — accuracy/precision/recall/F1 for Logistic Regression
   and Random Forest, plus a confusion matrix you can toggle between models.
5. **Interactive Prediction** — sliders/dropdowns for all raw features; the
   app applies the exact same feature engineering, encoding, scaling, and
   PCA transform used in training, then returns a live prediction and
   probability from your chosen model.
