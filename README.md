Stable Rankings, Shifting Fairness

Code for the paper "Stable Rankings, Shifting Fairness: What SHAP-Based Monitoring Can and Cannot Detect under Real-World Distribution Shift" by Partow Moradi Khushkanab (FH JOANNEUM, Graz, Austria).

📄 Paper: arXiv link coming soon

Overview

Machine learning models used in public services are often trained once and then reused across regions and years. Auditors frequently use SHAP feature attributions to check whether a deployed model still behaves as intended, because attributions can be computed without waiting for outcome labels.

This repository tests which label-free SHAP signals actually reveal changes in group fairness under naturally occurring distribution shift. Using the ACSPublicCoverage task from Folktables (US Census data), models are trained on one state-year and evaluated on 49 other state-years. Three signals are compared:

Rank signal: did the top-5 most important features change?
Global magnitude signal: how much did the mean absolute SHAP values change?
Group disparity signal: how much did the difference in SHAP values between demographic groups change?
Key findings
Rankings stay stable while fairness shifts. The top-5 features were unchanged in 84% of shifts, even when the false positive rate gap between groups changed by up to 13.4 percentage points.
Magnitudes track fairness drift. The change in global SHAP magnitudes correlated strongly with fairness change (mean Spearman ρ up to 0.78).
Group-level signals were weaker. Stratified sampling showed that sampling noise explains part, but not all, of this gap.
Sample size matters. Use several hundred explained instances per demographic cell.
Repository structure
File	Description
experiment_full.py	Experiment 1: 3 source states × 2 models (XGBoost, random forest) × 3 seeds × 49 target shifts × 3 group definitions
experiment_balanced.py	Experiment 2: stratified SHAP sampling (125–1000 samples per race × sex cell), XGBoost only
requirements.txt	Python dependencies
Installation
bash
git clone https://github.com/PartowMoradiK/stable-rankings-shifting-fairness.git
cd stable-rankings-shifting-fairness
pip install -r requirements.txt

The census data is downloaded automatically by Folktables on the first run.

Running the experiments
bash
python experiment_full.py        # Experiment 1 (about 2.5 hours on Google Colab)
python experiment_balanced.py    # Experiment 2 (about 20 minutes on Google Colab)

For a quick check that everything works, set QUICK_TEST = True at the top of each script (runs in a few minutes).

Both scripts save progress after every configuration. If a run is interrupted, running the script again resumes where it stopped.

Google Colab: run !pip install -q folktables xgboost shap fairlearn in the first cell, then paste the script into a second cell. To keep results after a disconnect, mount Google Drive and set OUT to a folder in your Drive.

Outputs

Experiment 1 (outputs/): all_results.csv, correlations.csv, summary.txt, table_main.tex and figures.

Experiment 2 (outputs_balanced/): balanced_results.csv, balanced_correlations.csv, balanced_summary.txt and figures.

Citation

If you use this code, please cite:

bibtex
@misc{moradikhushkanab2026stable,
  title  = {Stable Rankings, Shifting Fairness: What SHAP-Based Monitoring Can and Cannot Detect under Real-World Distribution Shift},
  author = {Moradi Khushkanab, Partow},
  year   = {2026},
  note   = {arXiv preprint}
}
License

This project is released under the MIT License.

Contact

Partow Moradi Khushkanab: partow.moradikhushkanab@edu.fh-joanneum.at
