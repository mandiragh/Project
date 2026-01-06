-- Procedures Exploratory Aanalysis --
-- Using procedure and d_items tables --

-- Joining procedure table to labels in d_items 
SELECT 
pe.itemid, 
di.label,
di.category, 
COUNT(*) AS N_EVENTS 
FROM mimic.procedureevents AS pe
LEFT JOIN mimic.d_items AS di
ON pe.itemid = di.itemid
GROUP BY pe.itemid, di.label, di.category
ORDER BY N_EVENTS DESC
LIMIT 50;  

-- Procedure Events by ICU Stays 
SELECT stay_id, 
COUNT (*) AS procedure_events
FROM mimic.procedureevents
GROUP BY stay_id;


-- Top procedure category per ICU Units 
SELECT 
i.first_careunit,
di.category,
COUNT (*) AS n_events
FROM mimic.procedureevents pe
JOIN mimic.icustays i
ON pe.stay_id = i.stay_id
LEFT JOIN mimic.d_items di
ON pe.itemid = di.itemid
GROUP BY i.first_careunit, di.category
ORDER BY i.first_careunit, n_events DESC;

---------------------------------------------------------------------
----------------------------------------------------------------------


-- Output events Exploratory Aanalysis --

-- Output item types 

SELECT 
oe.itemid,
di.label,
di.category,
COUNT(*) AS n_events
FROM mimic.outputevents oe
LEFT JOIN mimic.d_items di
ON oe.itemid = di.itemid 
GROUP BY oe.itemid , di.label, di.category
ORDER BY n_events DESC
LIMIT 50;

-- Total fluid Output per ICU stay 

SELECT
  stay_id,
  SUM(value) AS total_output_ml,
  COUNT(*) AS output_events
FROM mimic.outputevents
WHERE value IS NOT NULL
GROUP BY stay_id;


-----------------------------------------------------------------------
---------------------------------------------------------------------

