import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # processed klasöründen okur
    df = pd.read_csv("../../data/processed/processed_audiogram_data.csv")
    X = df[['Frequency (Hz)', 'Ear Side', 'Gender', 'Age', 'Air Conduction', 'Bone Conduction']]
    y = df[['Target 50 (db)', 'Target 65 (db)', 'Target 80 (db)', 'MPO']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Frequency (Hz)', 'Age', 'Air Conduction', 'Bone Conduction']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Ear Side', 'Gender'])
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    model = Sequential([
        Input(shape=(X_train_processed.shape[1],)),
        Dense(128, activation='relu'), Dropout(0.2),
        Dense(64, activation='relu'), Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(4, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint('../../models/best_nal_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
    ]

    model.fit(X_train_processed, y_train.values, validation_split=0.2, epochs=150, batch_size=32, callbacks=callbacks,
              verbose=1)

    predictions = model.predict(X_test_processed)
    logger.info(f"Yapay Sinir Ağı R^2 Skoru: {r2_score(y_test.values, predictions):.4f}")