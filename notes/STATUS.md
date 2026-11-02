# STATUS — ops-delivery-risk

**2026-11-02:** v1 complete and pushed (github.com/OmarDiaa5/ops-delivery-risk, CI green: python, dotnet, docker). Built to close the Lamdax JD gap (.NET, monitoring, operational data); resume at resumes/claude-resume-kit/output/Lamdax/.

Headline numbers (all verified by running): test ROC AUC 0.702, 23.2% of late orders caught in top 10% vs 12.5% for the seller-history rule (1.9x); late orders 2.57 vs 4.29 stars; replay of 32,150 orders: score PSI 0.19 → 0.93, flagged share 18% → 36%, monthly AUC 0.77 → 0.58; scoring 161 → 8.2 ms.

Local state: data/ (raw CSVs + ops.db, ~236 MB, gitignored, rebuild with build_features.py). Docker is not installed locally; images are only built in CI. mlflow.db is local only.

Possible next steps (only if a JD needs them): host the gateway + API on a free tier; add a retrain job triggered by the PSI alarm; calibrate scores (isotonic) so they read as probabilities.
