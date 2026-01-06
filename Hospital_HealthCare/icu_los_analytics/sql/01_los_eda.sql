-- -- Exploratory Data Analysis -- --
	-- ICU Stays-- 

-- Total Icu Stays 
SELECT COUNT (*) AS n_rows FROM mimic.icustays;

-- Unique ICU stays 
SELECT COUNT(DISTINCT stay_id) AS n_unique_stays FROM mimic.icustays;

-- ICU stays per patient 
SELECT 
COUNT(DISTINCT subject_id) AS n_patients, 
COUNT(*):: float / COUNT(DISTINCT subject_id) AS avg_stays_per_patient
FROM mimic.icustays;

---------------------------------------------------------------------
----------------------------------------------------------------------

-- Basic LOS profile
SELECT 
MIN(los) AS min_los, 
AVG(los) AS avg_los,
MAX(los) AS max_los
FROM mimic.icustays;

-- LOS by ICU Units 
SELECT 
first_careunit,
COUNT(*) AS stays,
AVG(los) AS avg_los,
MIN(los) AS min_los, 
MAX(los) AS max_los
FROM mimic.icustays
GROUP BY first_careunit
ORDER BY stays DESC;


-- Identify Outliers
WITH ranked AS (
SELECT 
stay_id, subject_id, first_careunit, los,
NTILE(100) OVER (ORDER BY los) AS pct_bucket
FROM mimic.icustays
)

SELECT * FROM ranked 
WHERE pct_bucket = 100
ORDER BY los DESC;
---------------------------------------------------------------------
----------------------------------------------------------------------

