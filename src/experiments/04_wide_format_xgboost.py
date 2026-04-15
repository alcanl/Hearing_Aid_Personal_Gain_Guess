import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_wide_features(filepath):
    logger.info("Temiz veri okunuyor ve Wide-Format (Geniş Format) dönüşümü başlatılıyor...")
    df = pd.read_csv(filepath)

    ac_wide = df.pivot_table(index=['ID', 'Ear Side'], columns='Frequency (Hz)', values='Air Conduction', aggfunc='first').reset_index()
    ac_wide.columns = ['ID', 'Ear Side'] + [f'AC_{int(col)}' for col in ac_wide.columns if col not in ['ID', 'Ear Side']]

    bc_wide = df.pivot_table(index=['ID', 'Ear Side'], columns='Frequency (Hz)', values='Bone Conduction', aggfunc='first').reset_index()
    bc_wide.columns = ['ID', 'Ear Side'] + [f'BC_{int(col)}' for col in bc_wide.columns if col not in ['ID', 'Ear Side']]

    logger.info("Tüm frekans verileri ana matrise entegre ediliyor...")
    df_wide = pd.merge(df, ac_wide, on=['ID', 'Ear Side'], how='left', validate="many_to_many")
    df_wide = pd.merge(df_wide, bc_wide, on=['ID', 'Ear Side'], how='left', validate="many_to_many")

    numeric_cols = df_wide.select_dtypes(include=[np.number]).columns
    df_wide[numeric_cols] = df_wide[numeric_cols].fillna(df_wide[numeric_cols].median())

    return df_wide

def run_wide_xgboost_experiment():
    df_wide = build_wide_features("../../data/processed/processed_audiogram_data.csv")

    ac_cols = [col for col in df_wide.columns if col.startswith('AC_')]
    bc_cols = [col for col in df_wide.columns if col.startswith('BC_')]

    numeric_features = ['Frequency (Hz)', 'Age', 'Air Conduction', 'Bone Conduction'] + ac_cols + bc_cols
    categorical_features = ['Ear Side', 'Gender']

    X = df_wide[numeric_features + categorical_features]
    y = df_wide[['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    xgb_estimator = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    multi_target_xgb = MultiOutputRegressor(xgb_estimator)

    ml_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', multi_target_xgb)
    ], memory=None)

    logger.info("Wide-Format XGBoost modeli eğitiliyor...")
    ml_pipeline.fit(X_train, y_train)

    predictions = ml_pipeline.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    logger.info(f"Wide-Format XGBoost R^2 Skoru: {r2:.4f}")
    logger.info(f"Ortalama Mutlak Hata (MAE): {mae:.4f} dB")

if __name__ == "__main__":
    run_wide_xgboost_experiment()