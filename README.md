


<div align="center">

  <h1>ANNITIA: risk stratification of metabolic dysfunction-associated steatotic liver disease (MASLD) through AI</h1>

  <br>

  <h4>
    • <a href="https://app.trustii.io/datasets/1551" target="_blank">[Challenge page]</a> •
  </h4>

  <p>
  <strong>⚠️ Please sign up first to access the challenge.</strong>
  </p>


</div>


The objective is to develop a risk stratification score to predict liver events related to the progression of MASLD based on non-invasive tests (NITs) repeated over time. Understanding how the trajectories of repeat NITs are associated with the disease progression is crucial for improving the current epidemiological surveillance for MASLD and for better personalizing the patient's follow-up healthcare.

This project is supported by France 2030 in partnership with the Health Data Hub.

## The vision behind ANNITIA

Metabolic dysfunction-associated steatotic liver disease (MASLD) is characterized by excessive fat accumulation in the liver that occurs without high alcohol consumption and affects approximately 30% of the world's population. The spectrum of the disease ranges from simple steatosis to more advancing forms, including steatohepatitis (MASH steatosis accompanied by inflammation and ballooning), that when persistent leads to the formation of scar tissue, known a fibrosis. Liver fibrosis is graded on a scale from F0 to F4 indicating, respectively, the absence of fibrosis to the most advanced stage  cirrhosis. The final stage of liver fibrosis (F4) increases the risk of liver-related events, such as ascites (fluid in the abdomen), jaundice (yellowing of the skin/eyes), or hepatic encephalopathy, esophageal varices (bleeding), hepatocellular carcinoma, liver transplantation, as well as death.

![Spectrum](images/masld_spectrum.png)


Therefore, liver fibrosis is a key factor contributing to poor outcomes in patients with MASLD. Detecting fibrosis in its earlier stages offers the best chance to slow, stop, or even reverse damage. Additionally, MASLD is closely linked not only to liver-specific complications but also to a wide range of cardiometabolic comorbidities, notably type 2 diabetes, elevated blood pressure and dyslipidemia. Thus, early identification of patients at risk of MASLD progression is critical for preventing long-term hepatic and cardiometabolic complications.

Currently, however, early detection of forms of steatohepatitis with advanced fibrosis remains a major clinical challenge, particularly in large cohorts of patients with metabolic comorbidities. This highlights the importance of developing more effective risk stratification tools before severe liver complications arise.

Furthermore, the progression of MASLD can vary from one patient to another. Therefore, early detection of patients at risk of progressing to more advanced stages of the disease using NIBTs is a crucial area of research in hepatology. Although liver histology is recognized as the gold standard for the definitive diagnosis of MASH and fibrosis, its invasive and costly nature, as well as its inter- and intra-observer variability, has always been an obstacle to epidemiological studies and therapeutic trials. It is therefore of the utmost importance to understand the value of NITs as a complementary tool or replacement for liver biopsy.

The diagnostic performance of NITs correlates with histological stages of fibrosis in cross-sectional settings, and they do not detect progressive forms of metabolic steatohepatitis. The focus is now on understanding how the trajectories of repeated NITs measurements over time are associated with disease progression or regression, as well as predicting the occurrence of long-term liver-related clinical events. Consequently, the development of NITs that could better identify patients at risk of MASLD progression would allow for better targeting of these patients with more systematic screening and more effective management, while also enabling modeling of the natural history of the disease.

## The Challenge: liver-related events predicted through Non-Invasive Tests (NIT)

The goal of the ANNITIA challenge is to use clinical data and non-invasive tests (NITs), such as FibroScan, FibroTest, and Aixplorer, to develop a score able to predict the occurrence of major liver events or death during the follow-up of patients with MASLD.

Model submissions are evaluated based on survival prediction performance using the Concordance Index (C-index). This metric assesses the model's ability to correctly rank patients according to their predicted risk relative to their observed event times.

Two clinical endpoints are considered:

Death event: weighted at 0.3 (30%).
Hepatic event: weighted at 0.7 (70%).

The final evaluation score is calculated as the weighted average of the two endpoint-specific C-indices:

Final Score=0.3×Cdeath+0.7×Chepatic

For each endpoint, the model must generate an individual risk score for every patient. These scores are evaluated against the observed survival outcomes and event times using the concordance_index_censored function from the scikit-survival library. A higher C-index indicates better discrimination between patients, reflecting a greater ability to correctly rank individuals according to their risk of experiencing the event.

The competitors will have access to data from 1,676 synthetic patients. The dataset includes:

Clinical data: age at baseline, gender, BMI (during follow-up), and medical history (diabetes, hypertension, dyslipidemia and bariatric surgery).
Biochemical parameters (from baseline to follow-up): repeated measurements of ALT, AST, GGT, bilirubin, platelets, glycemia, triglycerides, total-cholesterol.
Biomarkers (NIT, from baseline to follow-up): repeated measurements of FibroTest, FibroScan and Aixplorer.
Outcomes: occurrence of major liver events and all-cause mortality.

```mermaid
flowchart TD
    A["Raw longitudinal data"] --> B["Preprocessing and feature engineering"]

    B --> C{"Prediction task"}

    C -->|Death| D["Death prediction model"]
    C -->|Hepatic event| E["Hepatic event prediction model"]

    D --> F["Patient-level death risk scores"]
    E --> G["Patient-level hepatic risk scores"]

    F --> H["C-index death (0.3)"]
    G --> I["C-index hepatic (0.7)"]

    H --> J["Weighted evaluation"]
    I --> J

    J --> K["Final Score = 0.3 × C-index death + 0.7 × C-index hepatic"]

    classDef input fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef model fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef evaluation fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef output fill:#dcfce7,stroke:#16a34a,color:#14532d

    class A,B input
    class D,E,F,G model
    class H,I,J evaluation
    class K output
```


### Building a comprehensive dataset

A critical component of this challenge is the creation of a large dataset of patients with MASLD including a long-term follow-up (from 2001 to 2025) containing repeated measurements of different NITs over time.

The ANNITIA team has explored the clinical records of thousands of patients (who allowed it) at Pitié-Salpêtrière Hospital in Paris. In order to protect patients from any reidentification risks, synthetic data were created from the original clinical database. Thus, the following dataset features for each synthetical patient are available:

22 data points describing the patient over time of each examen performed, with the corresponding age, BMI and biochemical parameters.
Up to 3 NITs usually measured for assessing fibrosis progression.
One outcome column summarizing the liver-related events (composite endpoint) occurred by the patient during the follow-up and the time of the diagnosis: to be predicted.
One outcome column informing the death status (all-cause mortality): to be predicted.

This extended dataset provides an ideal foundation for training and evaluating machine learning models, offering participants a great opportunity to work with synthetic data generated from real-world medical data.


## Requirements

```
python -m venv annitia
source annitia/bin/activate
pip install -r requirements.txt
```

## Data

Please check out the [following page](https://app.trustii.io/datasets/1551) to download data. 


## Method

### Preprocessing

The preprocessing pipeline is designed to transform longitudinal clinical and biological data into patient-level features suitable for downstream predictive modelling. It accounts for repeated measurements collected across multiple visits and generates summary features capturing patients' clinical status, disease progression, and follow-up characteristics.

Feature engineering is performed separately depending on the prediction task (hepatic or death). 
**Longitudinal data processing**

The first step consists of extracting relevant information from repeated measurements collected during patient follow-up. Several patient-level summary features are generated, including:

Follow-up characteristics: number of available visits, last observed age, first and last available visits for each biomarker, and age differences between consecutive visits.
Measurement availability: number of non-missing measurements for each longitudinal variable and average number of measurements per visit.
Biomarker evolution: minimum, maximum, range, and mean differences between successive measurements.
Longitudinal trends: percentage changes relative to baseline and individual biomarker progression slopes estimated using the robust Theil-Sen method.

**Clinical and biological feature engineering**

Additional clinically relevant features are derived from longitudinal laboratory measurements and clinical characteristics:

Metabolic syndrome proxy score: composite score based on BMI, fasting glucose, triglycerides, hypertension, type 2 diabetes, and dyslipidaemia.
ALT/AST ratio: calculation of the ALT-to-AST ratio at each available visit.
Fibrosis staging: conversion of continuous FibroTest scores into categorical fibrosis stages (F0–F4), followed by the extraction of maximum fibrosis stage and advanced fibrosis indicators.
Cirrhosis proxy score: composite score based on bilirubin, platelet count, GGT, liver stiffness, and FibroTest measurements.

**Task-specific feature engineering**

Two distinct feature-engineering strategies are implemented according to the prediction objective:

Hepatic events prediction (hep): focuses on clinical and biological status, fibrosis severity, cirrhosis-related indicators, and the most recent available biomarker measurements. The original longitudinal biomarker columns are retained during the extraction of the latest measurements.

Mortality prediction (death): incorporates additional longitudinal information, including the number of available measurements, percentage changes from baseline, Theil-Sen progression slopes, and the number of examination intervals shorter than or equal to two years. The original longitudinal biomarker columns are removed after extracting the most recent available measurements.

**Final preprocessing**

Finally, categorical variables, including gender, type 2 diabetes, hypertension, dyslipidaemia, and bariatric surgery, are converted into binary dummy variables using one-hot encoding (drop_first=True). The resulting datasets contain engineered patient-level features tailored to each prediction task and are ready for subsequent modelling.

The complete preprocessing workflow is implemented in *[`preprocessing.py`](here).*

```mermaid
flowchart TD
    A["Raw longitudinal clinical and biological data"] --> B["Input DataFrame"]
    B --> C["Patient-level summary features"]
    C --> D{"Prediction task"}

    D -->|Hepatic events| H1["Clinical and longitudinal features"]
    D -->|Mortality| M1["Longitudinal measurement counts"]

    H1 --> H2["First and last biomarker visits"]
    H2 --> H3["Age differences between visits"]
    H3 --> H4["Metabolic syndrome proxy score"]
    H4 --> H5["ALT/AST ratios"]
    H5 --> H6["Fibrosis staging from FibroTest"]
    H6 --> H7["Cirrhosis proxy score"]
    H7 --> H8["Last observed age"]
    H8 --> H9["Extract most recent biomarker measurements"]

    M1 --> M2["Percentage changes from baseline"]
    M2 --> M3["Theil-Sen individual progression slopes"]
    M3 --> M4["First and last biomarker visits"]
    M4 --> M5["Age differences between visits"]
    M5 --> M6["Examination intervals within 2 years"]
    M6 --> M7["Fibrosis staging from FibroTest"]
    M7 --> M8["Cirrhosis proxy score"]
    M8 --> M9["Last observed age"]
    M9 --> M10["Extract most recent biomarker measurements"]

    H9 --> E["Categorical variable encoding"]
    M10 --> E

    E --> F["Final feature-engineered dataset"]
    F --> G["Downstream predictive modelling"]

    classDef input fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef process fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef decision fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef output fill:#dcfce7,stroke:#16a34a,color:#14532d

    class A,B input
    class C,H1,H2,H3,H4,H5,H6,H7,H8,H9,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,E process
    class D decision
    class F,G output
```

### Training 

To train our model,

```
python train.py
```

It applies our customized preprocessing (new features, change scores, summary metrics) , then trains and validate a RandomSurvivalForest using a robust validation strategy : nested cross-validation.

It prints ..., submits..




## Results

Hepatic nested CV: 0.8664 ± 0.0986
Death nested CV:   0.9473 ± 0.0266

