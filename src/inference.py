import joblib
import pandas as pd
import numpy as np


def predict_full_audiogram(patient_profile):
    print("Model Yükleniyor...")
    model = joblib.load('../models/biyonix_nal_model.pkl')

    audiogram = patient_profile['Audiogram']
    frequencies = list(audiogram.keys())

    pta_values = [audiogram.get(f, {}).get('AC', 0) for f in [500, 1000, 2000] if f in audiogram]
    pta = sum(pta_values) / len(pta_values) if pta_values else 0

    wide_ac_features = {f"AC_{freq}": data['AC'] for freq, data in audiogram.items()}

    rows = []
    for freq in frequencies:
        row = {
            'Frequency (Hz)': freq,
            'Age': patient_profile['Age'],
            'Ear Side': patient_profile['Ear Side'],
            'Gender': patient_profile['Gender'],
            'Air Conduction': audiogram[freq]['AC'],
            'Bone Conduction': audiogram[freq].get('BC', audiogram[freq]['AC']),
            'PTA': pta
        }
        row.update(wide_ac_features)
        rows.append(row)

    df_patient = pd.DataFrame(rows)

    standard_freqs = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    for sf in standard_freqs:
        col_name = f'AC_{sf}'
        if col_name not in df_patient.columns:
            df_patient[col_name] = df_patient['Air Conduction']

    print("Hedefler Hesaplanıyor...\n")
    predictions = model.predict(df_patient)

    results = {}
    for i, freq in enumerate(frequencies):
        results[freq] = {
            "Target 50": round(predictions[i][0], 1),
            "Target 65": round(predictions[i][1], 1),
            "Target 80": round(predictions[i][2], 1),
            "MPO": round(predictions[i][3], 1)
        }
    return results


if __name__ == "__main__":
    hasta_profili = {
        'Age': 65,
        'Gender': 'Male',
        'Ear Side': 'Left',
        'Audiogram': {
            250: {'AC': 30, 'BC': 25}, 500: {'AC': 45, 'BC': 40}, 750: {'AC': 50, 'BC': 45},
            1000: {'AC': 55, 'BC': 50}, 1500: {'AC': 60, 'BC': 55}, 2000: {'AC': 65, 'BC': 60},
            3000: {'AC': 70, 'BC': 65}, 4000: {'AC': 75, 'BC': 70}, 6000: {'AC': 80, 'BC': 75},
            8000: {'AC': 85, 'BC': 80}
        }
    }

    hedefler = predict_full_audiogram(hasta_profili)
    print(f"HASTA PROFİLİ: {hasta_profili['Age']} Yaş, {hasta_profili['Gender']}, {hasta_profili['Ear Side']}")
    print("-" * 65)
    for freq, t in hedefler.items():
        print(
            f"{freq:<6} Hz | T50: {t['Target 50']:<5} | T65: {t['Target 65']:<5} | T80: {t['Target 80']:<5} | MPO: {t['MPO']:<5}")