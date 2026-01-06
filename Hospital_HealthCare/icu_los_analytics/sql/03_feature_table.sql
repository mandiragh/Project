-- Feature table -- 
WITH proc AS (
  SELECT
    stay_id,
    COUNT(*) AS procedure_events
  FROM mimic.procedureevents
  GROUP BY stay_id
),
outp AS (
  SELECT
    stay_id,
    SUM(value) AS total_output_ml,
    COUNT(*) AS output_events
  FROM mimic.outputevents
  WHERE value IS NOT NULL
  GROUP BY stay_id
)
SELECT
  i.stay_id,
  i.subject_id,
  i.hadm_id,
  i.first_careunit,
  i.last_careunit,
  i.intime,
  i.outtime,
  i.los, 
  COALESCE(p.procedure_events, 0) AS procedure_events,
  COALESCE(o.total_output_ml, 0) AS total_output_ml,
  COALESCE(o.output_events, 0) AS output_events
FROM mimic.icustays i
LEFT JOIN proc p ON i.stay_id = p.stay_id
LEFT JOIN outp o ON i.stay_id = o.stay_id;

