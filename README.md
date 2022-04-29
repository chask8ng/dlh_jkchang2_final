# dlh_jkchang2_final
final project for dlh

The original paper "Automatic assignment of ICD codes based on clinical notes" can be found here: https://www.sciencedirect.com/science/article/pii/S1532046419300322#b0110

Following steps:

1. Data processing
Initial data processing step using MIMIC notes file (not included here due to access rules).
```
SELECT
  SUBJECT_ID,
  HADM_ID,
  REGEXP_REPLACE(REPLACE(SUBSTR(TEXT, STRPOS(TEXT, 'Discharge Diagnosis')),"\n", " "), "\[.*?\]", "") AS diagnosis_note
FROM
  `prefab-mile-346920.samples.parsed_notes` a
WHERE
  category = 'Discharge summary'
  AND CONTAINS_SUBSTR(TEXT,
    'Discharge Diagnosis')
```
This cleans up the notes to just target the actual diagnosis per visit, making it simpler to look for keywords.
With the cleaned notes, use word2vec to capture associations of keywords.
The keywords are then used to train/test a bi-directional LSTM model to predict the target ICD codes.

![Screen Shot 2022-04-29 at 7 44 54 AM](https://user-images.githubusercontent.com/87827828/165938562-de0f26cb-78dc-47e3-9ecb-9c566741613b.png)
