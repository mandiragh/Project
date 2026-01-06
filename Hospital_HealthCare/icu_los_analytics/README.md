

---

# 🏥 ICU Length of Stay & Resource Utilization Dashboard

**Operational Insights from MIMIC-III ICU Data**

---

## 📌 Project Overview

This project analyzes **Intensive Care Unit (ICU) length of stay (LOS)** and **resource utilization patterns** using the **MIMIC-III clinical database**. The goal is to identify **drivers of prolonged ICU stays**, understand **variability across care units**, and surface **operational insights** that can support **ICU capacity planning and healthcare decision-making**.

The analysis combines **SQL-based exploratory data analysis**, **feature engineering at the ICU stay level**, and an **interactive Tableau dashboard** to communicate findings clearly to non-technical stakeholders.

---

## 🎯 Key Questions Addressed

* How is ICU length of stay distributed, and where do long-stay outliers occur?
* How does ICU length of stay vary across different care units?
* Is higher procedural burden associated with prolonged ICU stays?
* How does fluid output (as a proxy for patient acuity) relate to ICU LOS?
* Are there temporal patterns in ICU admissions over time?
* What proportion of ICU stays fall into high-risk (long-stay) categories?

---

## 🗂️ Data Sources

This project uses publicly available, de-identified data from **MIMIC-III**, focusing on the following tables:

* **`icustays`** – ICU admission details, timestamps, care units, and length of stay
* **`procedureevents`** – ICU procedures performed during each stay
* **`outputevents`** – Recorded fluid outputs during ICU stays
* **`d_items`** – Dictionary table used to interpret event identifiers

All data is aggregated to a **stay-level dataset** to support operational analysis while preserving patient privacy.

---

## 🔧 Data Preparation & Feature Engineering

### SQL-based EDA included:

* ICU stay counts and patient distribution
* Length of stay statistics and outlier detection
* Procedure and output event frequency analysis
* Care unit comparisons

### Key engineered features (1 row per ICU stay):

* **Length of stay (days)**
* **Total procedure events per stay**
* **Total fluid output (mL)**
* **Care unit transfer flag** (transferred vs. no transfer)
* **Length-of-stay risk bands**:

  * Short Stay (< 1 day)
  * Medium Stay (1–3 days)
  * Long Stay (> 3 days)

The final dataset was exported as a clean, Tableau-ready CSV.

---

## 📊 Dashboard Overview (Tableau)
Link: https://public.tableau.com/views/ICULengthofStayResourceUtilizationDashboard/ICU?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link 

The interactive Tableau dashboard is structured to mirror how healthcare leaders evaluate ICU performance:

### KPI Summary

* Total ICU admissions
* Average ICU length of stay
* Average procedures per ICU stay
* Percentage of long-stay ICU patients

### Core Visualizations

* ICU length-of-stay distribution
* Average LOS by care unit
* Procedure burden vs. LOS
* Fluid output vs. LOS
* ICU admissions trend over time
* Distribution of ICU stays by LOS risk band

### Interactivity

* Filters by care unit, LOS risk band, transfer status, and LOS range
* Reference lines highlighting average and long-stay thresholds

---

## 📈 Key Insights

* ICU length of stay is **right-skewed**, with a small proportion of stays accounting for a large share of ICU utilization.
* Average LOS varies meaningfully across ICU care units.
* Higher procedure intensity is associated with longer ICU stays.
* Increased fluid output correlates with prolonged ICU utilization, reflecting higher patient acuity.
* ICU admissions exhibit temporal patterns that may inform staffing and bed-planning decisions.
* A distinct subset of long-stay patients contributes disproportionately to resource consumption.

---

## 🧠 Tools & Technologies

* **SQL** – Exploratory analysis and feature engineering
* **Tableau** – Interactive dashboard and data visualization
* **Python (optional)** – Data inspection and export
* **MIMIC-III Database** – De-identified clinical data

---

## 📁 Repository Structure

```
icu-los-resource-utilization/
│
├── sql/
│   ├── 01_los_eda.sql
│   ├── 02_procedure_analysis.sql
│   ├── 03_output_analysis.sql
│   └── 04_stay_level_features.sql
│
├── data/
│   ├── processed/
│   │   └── icu_stay_features.csv
│
├── dashboards/
│   └── icu_los_resource_utilization.png
│
└── README.md
```

---

## 🚀 Future Enhancements

* Incorporate diagnoses and comorbidity data for clinical risk adjustment
* Add early-ICU (first 24 hours) indicators for predictive modeling
* Extend analysis to ICU readmissions and mortality outcomes
* Build a predictive model for prolonged ICU stays

---

## 📜 Disclaimer

This project uses **de-identified clinical data** for educational and analytical purposes only.
It does **not** provide medical advice or clinical recommendations.

---

## 👤 Author

**Mandira Ghimire**
MS in Data Analytics






