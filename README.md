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

Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 2980)]       0           []                               
                                                                                                  
 embedding (Embedding)          (None, 2980, 100)    3357300     ['input_1[0][0]']                
                                                                                                  
 permute (Permute)              (None, 100, 2980)    0           ['embedding[0][0]']              
                                                                                                  
 dense (Dense)                  (None, 100, 2980)    8883380     ['permute[0][0]']                
                                                                                                  
 attention (Permute)            (None, 2980, 100)    0           ['dense[0][0]']                  
                                                                                                  
 multiply (Multiply)            (None, 2980, 100)    0           ['embedding[0][0]',              
                                                                  'attention[0][0]']              
                                                                                                  
 bidirectional (Bidirectional)  (None, 2980, 10)     4240        ['multiply[0][0]']               
                                                                                                  
 bidirectional_1 (Bidirectional  (None, 10)          640         ['bidirectional[0][0]']          
 )                                                                                                
                                                                                                  
 dense_1 (Dense)                (None, 64)           704         ['bidirectional_1[0][0]']        
                                                                                                  
 dense_2 (Dense)                (None, 5)            325         ['dense_1[0][0]']                
                                                                                                  
==================================================================================================
Total params: 12,246,589
Trainable params: 8,889,289
Non-trainable params: 3,357,300

