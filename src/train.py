import logging
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_and_engineer_features(in_path, out_path):
    logger.info("Ham dosyalar okunuyor ve Domain Engineering başlıyor...")
    df_in = pd.read_excel(in_path)
    df_out = pd.read_excel(out_path)

    df_out_gen = df_out[df_out['Program'] == 'General'].copy()
    for col in ['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']:
        df_out_gen[col] = pd.to_numeric(df_out_gen[col].replace({'???': np.nan, ' ???': np.nan, '140+': 140}), errors='coerce')

    df_out_grouped = df_out_gen.dropna(subset=['Target 50 (db)']).groupby(
        ['ID', 'Frequency (Hz)', 'Ear Side']).mean(numeric_only=True).reset_index()

    df_in_piv = df_in.pivot_table(index=['ID', 'Frequency (Hz)', 'Ear Side', 'Gender', 'Age'],
                                  columns='Audiogram Type', values='Decibel (dB)').reset_index()

    df_merged = pd.merge(df_in_piv, df_out_grouped, on=['ID', 'Frequency (Hz)', 'Ear Side'], how='inner')

    ac_wide = df_merged.pivot_table(index=['ID', 'Ear Side'], columns='Frequency (Hz)',
                                    values='Air Conduction', aggfunc='first').reset_index()
    ac_wide.columns = ['ID', 'Ear Side'] + [f'AC_{int(col)}' for col in ac_wide.columns if col not in ['ID', 'Ear Side']]
    df_wide = pd.merge(df_merged, ac_wide, on=['ID', 'Ear Side'], how='left')

    logger.info("PTA (Saf Ses Ortalaması) hesaplanıyor...")
    pta_cols = ['AC_500', 'AC_1000', 'AC_2000']
    if all(col in df_wide.columns for col in pta_cols):
        df_wide['PTA'] = df_wide[pta_cols].mean(axis=1)
    else:
        df_wide['PTA'] = df_wide['Air Conduction']

    numeric_cols = df_wide.select_dtypes(include=[np.number]).columns
    df_wide[numeric_cols] = df_wide[numeric_cols].fillna(df_wide[numeric_cols].median())
    df_wide = df_wide.dropna(subset=['Target 50 (db)'])

    return df_wide

if __name__ == "__main__":
    df = preprocess_and_engineer_features("../data/raw/inputs_1.xlsx", "../data/raw/outputs_1.xlsx")

    ac_cols = [col for col in df.columns if col.startswith('AC_')]
    numeric_features = ['Frequency (Hz)', 'Age', 'Air Conduction', 'Bone Conduction', 'PTA'] + ac_cols
    categorical_features = ['Ear Side', 'Gender']

    X = df[numeric_features + categorical_features]
    y = df[['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    xgb_estimator = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
    ml_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', MultiOutputRegressor(xgb_estimator))])

    logger.info("Model eğitiliyor...")
    ml_pipeline.fit(X_train, y_train)

    preds = ml_pipeline.predict(X_test)
    logger.info(f"Nihai R^2 Skoru: {r2_score(y_test, preds):.4f}")
    logger.info(f"Ortalama Mutlak Hata (MAE): {mean_absolute_error(y_test, preds):.4f} dB")

    joblib.dump(ml_pipeline, '../models/biyonix_nal_model.pkl')
    logger.info("Model '../models/biyonix_nal_model.pkl' konumuna kaydedildi.")