# HeartRisk Studio Frontend

A lightweight React + Vite dashboard for exploring patient-level heart-risk estimates in the heart disease mortality project.

## Full Stack Architecture

- **Backend API**: Python Flask service running on `http://127.0.0.1:5000` with pre-cached Scikit-Learn models (`models/logistic.pkl`, `models/random_forest.pkl`, `models/gradient_boosting.pkl`).
- **Frontend**: React + Vite on `http://localhost:5173`, automatically reverse-proxying `/api` requests to port 5000.

## Run Full Stack

1. **Start Python API Backend**:
   ```bash
   python backend/server.py
   ```
2. **Start Frontend Dev Server**:
   ```bash
   cd frontend
   npm run dev
   ```

3. Open **http://localhost:5173** in your browser.
