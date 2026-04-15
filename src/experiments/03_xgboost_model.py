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

# --- LOGLAMA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_xgboost_experiment():
    logger.info("İşlenmiş veri seti yükleniyor...")
    # Yol experiments klasöründen dışarı çıkacak şekilde ayarlandı
    df = pd.read_csv("../../data/processed/processed_audiogram_data.csv")

    X = df[['Frequency (Hz)', 'Ear Side', 'Gender', 'Age', 'Air Conduction', 'Bone Conduction']]
    y = df[['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ['Frequency (Hz)', 'Age', 'Air Conduction', 'Bone Conduction']
    categorical_features = ['Ear Side', 'Gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    xgb_estimator = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    multi_target_xgb = MultiOutputRegressor(xgb_estimator)

    ml_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', multi_target_xgb)
    ])

    logger.info("Standart XGBoost algoritması ile eğitim başlıyor...")
    ml_pipeline.fit(X_train, y_train)

    predictions = ml_pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    logger.info(f"XGBoost (Standart) R^2 Skoru: {r2:.4f}")
    logger.info(f"Ortalama Mutlak Hata (MAE): {mae:.4f} dB")


if __name__ == "__main__":
    run_xgboost_experiment()