
# 1️⃣ Dashboard Description 

## Dashboard Title

**ICU Length of Stay & Resource Utilization Overview**

### Purpose

This dashboard provides an **operational overview of ICU utilization**, focusing on **length of stay (LOS)**, **procedural burden**, **fluid output**, and **temporal admission patterns** across ICU care units. It is designed to help identify **high-resource ICU stays** and **drivers of prolonged ICU utilization**.

---

## Top KPI Summary (Executive Snapshot)

At the top of the dashboard, four KPIs summarize ICU utilization at a glance:

* **Total ICU Admissions:** 94,458
* **Average ICU Length of Stay:** 3.63 days
* **Average Procedures per ICU Stay:** 8.56
* **Percentage of Long-Stay ICU Patients:** 33.01%

These KPIs indicate that **one-third of ICU stays exceed 3 days**, suggesting a substantial long-stay burden on ICU capacity and resources.

---

## Length of Stay Distribution (Top Left)

The **LOS distribution histogram** shows a **strong right-skew**:

* The majority of ICU stays are **short (< 2 days)**
* A smaller subset of stays extends to **very long durations**, forming a long tail

This highlights that **a relatively small proportion of patients account for a large share of ICU bed-days**, a common challenge in ICU operations.

---

## Average LOS by Care Unit (Top Right)

The bar chart comparing **average LOS across ICU care units** reveals substantial variation:

* Neurology and specialty surgical units show **significantly higher average LOS**
* Medical and mixed medical-surgical units cluster around the overall average
* Stepdown and coronary units generally have shorter stays

This variation suggests **unit-specific workflows, patient complexity, and care pathways** strongly influence ICU length of stay.

---

## Procedure Burden vs Length of Stay (Middle Left)

This scatter plot shows the relationship between **procedural intensity** and **average LOS**:

* Units with **higher procedure counts** tend to have **longer average LOS**
* High-intensity units cluster toward the upper-right region of the chart
* Bubble size reflects output volume, reinforcing the association between **clinical complexity and resource use**

This indicates that **procedural burden is a key driver of prolonged ICU stays**.

---

## Fluid Output vs Length of Stay (Middle Center)

This scatter plot examines **total fluid output** as a proxy for patient acuity:

* Higher fluid output is associated with **longer ICU stays**
* Several care units exceed the overall LOS reference line while also exhibiting high output volumes

This suggests that **physiological instability and acuity** are closely linked with prolonged ICU utilization.

---

## Long Stay Based on Output & Procedures (Middle Right)

This paired bar chart focuses on **long-stay ICU patients only**, comparing:

* **Average procedure events**
* **Average total fluid output**

across care units.

The visualization shows that **long-stay patients consistently require more procedures and generate higher output volumes**, reinforcing their disproportionate impact on ICU resources.

---

## ICU Admissions by Month (Bottom Left)

The line chart displays **monthly ICU admissions by care unit**:

* Admissions remain relatively **stable throughout the year**
* Mild seasonal variation is observed in some units
* No extreme spikes, indicating consistent baseline ICU demand

This supports **predictable staffing and capacity planning**, rather than reactive surge management.

---

## Risk Band Distribution (Bottom Center)

The LOS risk band chart categorizes ICU stays into:

* **Short Stay (< 1 day)**
* **Medium Stay (1–3 days)**
* **Long Stay (> 3 days)**

Key observation:

* **Medium and Long stays dominate total ICU utilization**
* Long stays represent a smaller count but a **significant operational burden**

---
---

## 📊 Key Findings & Insights

### 1. ICU Length of Stay Is Highly Skewed

ICU length of stay exhibits a strong right-skewed distribution. While most patients spend less than two days in the ICU, a small subset of long-stay patients contributes disproportionately to total ICU bed utilization.

**Operational implication:**
Targeting long-stay patients offers the greatest opportunity for improving ICU throughput.

---

### 2. One-Third of ICU Patients Are Long-Stay

Approximately **33% of ICU patients remain in the ICU for more than three days**. This indicates that prolonged stays are not rare edge cases but a core operational challenge.

**Operational implication:**
Capacity planning should explicitly account for long-stay patient cohorts rather than assuming rapid turnover.

---

### 3. Significant Variation Exists Across Care Units

Average ICU length of stay varies widely by care unit, with neurology and specialized surgical units exhibiting the longest stays. In contrast, coronary and stepdown units show shorter and more predictable stays.

**Operational implication:**
Unit-specific performance benchmarks may be more effective than system-wide LOS targets.

---

### 4. Procedural Burden Is Strongly Associated With Prolonged LOS

Care units performing higher numbers of procedures per stay consistently show longer average LOS. Long-stay ICU patients also undergo substantially more procedures than short-stay patients.

**Operational implication:**
Early identification of high-procedure patients could enable proactive resource allocation and discharge planning.

---

### 5. Fluid Output Reflects Patient Acuity and Resource Intensity

Higher total fluid output correlates with longer ICU stays, supporting its use as a proxy for patient acuity. Long-stay patients generate markedly higher output volumes across multiple care units.

**Operational implication:**
Monitoring early output trends may help flag patients at risk of prolonged ICU stays.

---

### 6. ICU Admissions Show Stable Temporal Patterns

Monthly ICU admissions remain relatively stable throughout the year, with only modest seasonal fluctuations.

**Operational implication:**
ICU demand is largely predictable, enabling planned staffing models rather than crisis-driven responses.

---

### 7. Medium and Long Stays Dominate ICU Utilization

While short stays are common, medium (1–3 days) and long (>3 days) stays account for the majority of ICU utilization.

**Operational implication:**
Reducing LOS by even a small margin among medium- and long-stay patients could significantly improve bed availability.

---

## 📌 Summary Insight

> **A relatively small subset of high-acuity, high-procedure ICU patients drives a disproportionate share of ICU resource utilization. Identifying and managing these patients early represents the greatest opportunity for improving ICU efficiency and capacity.**

