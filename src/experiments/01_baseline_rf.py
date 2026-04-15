import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudiogramDataProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def process_data(self):
        inputs_df = pd.read_excel(self.input_path)
        outputs_df = pd.read_excel(self.output_path)
        outputs_df = outputs_df.replace({'???': np.nan, ' ???': np.nan, '140+': 140})

        target_cols = ['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']
        for col in target_cols:
            outputs_df[col] = pd.to_numeric(outputs_df[col], errors='coerce')

        outputs_df = outputs_df.dropna(subset=target_cols)
        out_gen = outputs_df[outputs_df['Program'] == 'General'].copy()

        in_pivot = inputs_df.pivot_table(index=['ID', 'Frequency (Hz)', 'Ear Side', 'Gender', 'Age'], columns='Audiogram Type', values='Decibel (dB)').reset_index()
        final_df = pd.merge(in_pivot, out_gen, on=['ID', 'Frequency (Hz)', 'Ear Side'], how='inner')

        final_df['Air Conduction'] = final_df['Air Conduction'].fillna(final_df['Air Conduction'].median())
        final_df['Bone Conduction'] = final_df['Bone Conduction'].fillna(final_df['Air Conduction'])

        # İşlenmiş veriyi processed klasörüne kaydet
        final_df.to_csv("../../data/processed/processed_audiogram_data.csv", index=False)
        return final_df

if __name__ == "__main__":
    # Yollar raw klasörüne göre uyarlandı
    processor = AudiogramDataProcessor("../../data/raw/inputs_1.xlsx", "../../data/raw/outputs_1.xlsx")
    df = processor.process_data()

    X = df[['Frequency (Hz)', 'Ear Side', 'Gender', 'Age', 'Air Conduction', 'Bone Conduction']]
    y = df[['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['Frequency (Hz)', 'Age', 'Air Conduction', 'Bone Conduction']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['Ear Side', 'Gender'])
    ])

    ml_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    ml_pipeline.fit(X_train, y_train)
    logger.info(f"Baseline Model R^2 Skoru: {ml_pipeline.score(X_test, y_test):.4f}")