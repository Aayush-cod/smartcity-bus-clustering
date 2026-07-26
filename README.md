🚌 Smart City Bus Clustering — Route Irregularity Detection for Greater Manchester

Geospatial Clustering of Urban Bus GPS Data to Identify Route Irregularities for Smart City Planning

A PySpark-based big data pipeline that ingests live bus GPS data, timetables, fares, and disruption records from the UK's Bus Open Data Service (BODS), and uses unsupervised clustering to detect behavioural irregularities — schedule deviation, abnormal speed, and disruption exposure — across the Greater Manchester Bee Network. Results are surfaced through an interactive Streamlit dashboard for use by an Urban Planner / Smart City Team.

📌 Project Summary
	
Stakeholder	Urban Planner / Smart City Team
Business problem	Understanding peak travel patterns and route irregularities for infrastructure decisions
ML category	Clustering (unsupervised)
Models compared	K-Means (selected), Gaussian Mixture Model, DBSCAN
Dataset size	771,733 combined rows (exceeds 100,000-record requirement)
Data sources	BODS: SIRI-VM (location), TXC (timetables), NeTEx (fares), disruption catalogue
Tech stack	PySpark, Spark MLlib, SQLite, scikit-learn, Streamlit, Plotly
🧠 What This System Does

Raw bus GPS data is just a stream of coordinates — on its own, it doesn't tell a planner anything actionable. This system turns ~770,000 raw GPS pings into 6 interpretable behavioural clusters, each describing a distinct pattern of schedule adherence, speed, and disruption exposure.

Headline finding: one cluster of buses (primarily operator BNSM) averages 282 minutes behind schedule — a severe, specific, investigable reliability problem that would be invisible in raw data but is immediately obvious once clustered.

🏗️ Architecture
BODS APIs (SIRI-VM, TXC, NeTEx, Disruptions)
        │
        ▼
 [Notebook 01] Ingestion & Combination
   XML parsing (distributed via mapPartitions) → broadcast joins → raw dataset
        │
        ▼
 [Notebook 02] Cleaning & Preprocessing
   Null handling, schema fixes, data quality audit
        │
        ▼
 [Notebook 03] Data Storage & Relationships
   Normalised SQLite schema (star schema: fact + 3 dimension tables)
        │
        ▼
 [Notebook 04] Exploratory Data Analysis
   Distributions, correlations, geospatial density, temporal patterns
        │
        ▼
 [Notebook 05] Feature Engineering
   Schedule deviation, speed (Haversine), disruption exposure,
   categorical encoding, VectorAssembler + StandardScaler
        │
        ▼
 [Notebook 06] Clustering Models
   K-Means / GMM / DBSCAN — CrossValidator, Silhouette Score, WCSS
        │
        ▼
 [Notebook 07] Evaluation & Model Selection
   Comparison table, complexity analysis, final model export
        │
        ▼
 [Streamlit Dashboard] Visualisation
   Interactive map, cluster profiles, delay rankings, model transparency
📂 Folder Structure
smartcity-bus-clustering/
├── data/
│   ├── raw/                  # Original BODS XML/CSV (not tracked — see note below)
│   ├── processed/            # Cleaned & feature-engineered datasets
│   ├── database/             # bus_clustering.db (SQLite)
│   └── models/               # Saved K-Means model + feature pipeline
├── notebooks/
│   ├── 01_build_raw_combined_dataset.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_data_storage_and_relationships.ipynb
│   ├── 04_eda_visualizations.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_clustering_models.ipynb
│   └── 07_evaluation.ipynb
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/config.toml
├── .gitignore
└── README.md
⚙️ Setup

Requirements: Python 3.13, Java 8+ (for Spark), ~16GB RAM recommended.

bash
git clone <this-repo-url>
cd smartcity-bus-clustering

# Notebook dependencies
pip install pyspark pandas numpy matplotlib seaborn scikit-learn --break-system-packages

# Dashboard dependencies
pip install -r dashboard/requirements.txt --break-system-packages
▶️ How to Run
Full pipeline (from raw data)

Run notebooks in order, 01 → 07, inside Jupyter Lab/Notebook. Each notebook is self-contained (re-establishes its own Spark session and reloads required data), so they can be run independently as long as the previous notebook's output already exists on disk.

bash
jupyter lab notebooks/

⚠️ Notebook 01 requires a valid BODS API key (free registration at data.bus-data.dft.gov.uk) to re-collect live GPS data. Static timetable/ fares/disruption catalogues are downloaded separately from BODS.

Dashboard only (fastest — uses pre-computed data)
bash
cd dashboard
streamlit run app.py

The dashboard reads data/processed/dashboard_data_final.csv, which is included in this repository, so it runs immediately without needing to execute any notebooks first.

📊 Model Comparison
Model	Silhouette Score	Training Time	Verdict
K-Means (k=6)	0.456	3.2s	✅ Selected — balanced, interpretable clusters
Gaussian Mixture (k=5)	0.274–0.307	15.4s	❌ Gaussian assumption violated by one-hot categorical features
DBSCAN	0.02–0.50*	3–6s (sampled)	❌ Best silhouette is a degenerate single-cluster collapse (~94% of points); best meaningful config scores only 0.17

*See Notebook 06 for the full eps tuning journey and diagnostic reasoning.

Complexity: K-Means O(n·k·i·d) vs GMM O(n·k·d²·i) — the quadratic dependency on feature dimensionality (d=13) directly explains GMM's ~5× slower runtime and lower accuracy on this mixed continuous/categorical feature space.

⚠️ Known Limitations
Collection window bias: location data was collected in bursts over ~33 hours (heavily evening/night-weighted), so hour-of-day and peak/off-peak features do not reflect true 24-hour ridership patterns.
BNML fares gap: one operator had no published fares dataset on BODS at collection time (~242k rows have null fare fields — a genuine data availability gap, not a processing error).
Disruption feature has low variance: most short-term disruptions overlapped the entire collection window, so disruption_count is a weak discriminating feature for clustering.
Timing/behaviour irregularity, not physical path deviation: this system detects schedule and speed anomalies, not whether a bus physically diverted from its assigned road route.
💾 Data Availability Note

Due to GitHub's 100MB file size limit, the following are not included:

data/raw/ — original BODS XML files (~3.4GB)
data/processed/cleaned_dataset_final.csv (260MB)
data/processed/features_dataset.parquet/

Included: dashboard_data_final.csv (74MB), the SQLite database, and the trained model — sufficient to run the dashboard and inspect results without regenerating the full pipeline. To regenerate excluded files, run notebooks 01–07 in order.

🎓 Coursework Context

Built for a Big Data / Data Science coursework module. Demonstrates distributed processing (PySpark, ≥8 partitions, caching, repartitioning, broadcast joins), relational database design with foreign-key relationships, injection-safe parameterised queries, EDA with statistical profiling, feature engineering, comparative ML model evaluation, and software delivery via an interactive dashboard.