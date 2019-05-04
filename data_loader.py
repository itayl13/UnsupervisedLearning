import pandas as pd
import os
import csv
import numpy as np


def load_data():
    # Load data and remove features that are missing for more than 50% of the rows.
    # Remove also rows that have other missing features.
    # Also, take only one row for every patient.
    data_file = os.path.join(os.getcwd(), 'dataset_diabetes', 'diabetic_data.csv')
    original_df = pd.read_csv(data_file)
    df_without_missing_ftrs = original_df.drop(['weight', 'payer_code', 'medical_specialty'], axis=1)
    missing_rows = []
    for row_idx, row in df_without_missing_ftrs.iterrows():
        if any([row[col] == '?' for col in df_without_missing_ftrs.columns]):
            missing_rows.append(row_idx)
    df_without_missing = df_without_missing_ftrs.drop(missing_rows, axis=0)
    df_unique = df_without_missing.drop_duplicates('patient_nbr', keep='first')
    df_unique.to_csv('ftr_matrix_with_nominals.csv', index=False)
    return df_unique


def build_diag_dict(icd9):
    # Create a dictionary of {icd9 code: diagnosis description} by the paper that describes the dataset.
    diag_dict = {}
    for code in icd9:
        try:
            c = float(code)
            if 0 < c <= 139:
                diag_dict[code] = 'infection_parasitic'
            elif 140 <= c <= 239:
                diag_dict[code] = 'neoplasms'
            elif 240 <= c <= 279 and c != 250:
                diag_dict[code] = 'metabolic'
            elif 250 <= c < 251:
                diag_dict[code] = 'diabetes'
            elif 280 <= c <= 289:
                diag_dict[code] = 'blood'
            elif 290 <= c <= 319:
                diag_dict[code] = 'mental_disorders'
            elif 320 <= c <= 359:
                diag_dict[code] = 'nervous_sys'
            elif 360 <= c <= 389:
                diag_dict[code] = 'sense_organs'
            elif 390 <= c <= 459 or c == 785:
                diag_dict[code] = 'circulatory_sys'
            elif 460 <= c <= 519 or c == 786:
                diag_dict[code] = 'respiratory_sys'
            elif 520 <= c <= 579 or c == 787:
                diag_dict[code] = 'digestive_sys'
            elif 580 <= c <= 629 or c == 788:
                diag_dict[code] = 'genitourinary_sys'
            elif 630 <= c <= 679:
                diag_dict[code] = 'pregnancy_childbirth'
            elif 680 <= c <= 709 or c == 782:
                diag_dict[code] = 'skin'
            elif 710 <= c <= 739:
                diag_dict[code] = 'musculoskeletal_sys'
            elif 740 <= c <= 759:
                diag_dict[code] = 'cogenital_anomalies'
            elif c in [780, 781, 784] + list(range(790, 800)):
                diag_dict[code] = 'other'
            elif 800 <= c <= 999:
                diag_dict[code] = 'injury_poisoning'
            elif c == 783:
                diag_dict[code] = 'endocrine_sys'
            elif c == 789:
                diag_dict[code] = 'abdomen_pelvis'
        except ValueError:
            diag_dict[code] = 'injury_external'
    return diag_dict


def nominals_to_numbers():
    # From the loaded DataFrame after removing missing and duplicates, set all the data to be numeric.
    if os.path.exists('ftr_matrix_with_nominals.csv'):
        df = pd.read_csv('ftr_matrix_with_nominals.csv')
    else:
        df = load_data()
    race_dict = {'AfricanAmerican': 0, 'Asian': 1, 'Caucasian': 2, 'Hispanic': 3, 'Other': 4}
    gender_dict = {'Female': 0, 'Male': 1, 'Unknown/Invalid': 2}
    age_dict = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5,
       '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
    change_dict = {'No': 0, 'Ch': 1}
    diabetes_med_dict = {'No': 0, 'Yes': 1}
    readmitted_dict = {'NO': 0, '<30': 1, '>30': 2}
    diag_dict = build_diag_dict(np.unique(df[['diag_1', 'diag_2', 'diag_3']]))

    # with open('nominals_dictionaries.csv', 'w') as f:
    #     w = csv.writer(f)
    #     w.writerow(['race'])
    #     w.writerow([race_dict])
    #     w.writerow(['gender'])
    #     w.writerow([gender_dict])
    #     w.writerow(['age'])
    #     w.writerow([age_dict])
    #     w.writerow(['Diag_1 to 3'])
    #     w.writerow([diag_dict])
    #     w.writerow(['Glucose_serum_test_result'])
    #     w.writerow(['dummy values'])
    #     w.writerow(['A1C_test_result'])
    #     w.writerow(['dummy values'])
    #     w.writerow(['Medications_features(metformin to metformin-pioglitazone)'])
    #     w.writerow(['dummy values'])
    #     w.writerow(['Change'])
    #     w.writerow([change_dict])
    #     w.writerow(['Taking_diabetes_medications'])
    #     w.writerow([diabetes_med_dict])
    #     w.writerow(['Readmitted'])
    #     w.writerow([readmitted_dict])



    old_columns = {colname: colidx for colidx ,colname in enumerate(df.columns)}
    df['race'] = df['race'].replace(race_dict)
    df['gender'] = df['gender'].replace(gender_dict)
    df['age'] = df['age'].replace(age_dict)
    df['change'] = df['change'].replace(change_dict)
    df['diabetesMed'] = df['diabetesMed'].replace(diabetes_med_dict)
    df['readmitted'] = df['readmitted'].replace(readmitted_dict)
    df[['diag_1', 'diag_2', 'diag_3']] = df[['diag_1', 'diag_2', 'diag_3']].replace(diag_dict)
    dummies_no_meds = pd.get_dummies(df[['diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult']])
    meds = range(old_columns['metformin'], old_columns['metformin-pioglitazone'] + 1)
    dummies_meds = pd.get_dummies(df[[col for col in old_columns.keys() if old_columns[col] in meds]])
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult'] +
                         [col for col in old_columns.keys() if old_columns[col] in meds])
    df = pd.concat([df[['patient_nbr', 'race', 'gender', 'age']], dummies_no_meds,
                   df.drop(columns=['patient_nbr', 'race', 'gender', 'age', 'change', 'diabetesMed', 'readmitted']),
                    dummies_meds, df[['change', 'diabetesMed', 'readmitted']]], axis=1)
    df = df.drop(columns=['encounter_id'])
    df.to_csv('final_ftr_matrix.csv', index=False)


if __name__ == "__main__":
    nominals_to_numbers()
