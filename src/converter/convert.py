import joblib
import pandas as pd
import numpy as np
from skl2onnx import to_onnx
import onnxruntime as rt
import re

from xgboost import XGBRegressor
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

print("ONNX dönüştürücü ve XGBoost eklentileri yükleniyor...")

update_registered_converter(
    XGBRegressor, 'XGBoostXGBRegressor',
    calculate_linear_regressor_output_shapes, convert_xgboost
)

pipeline = joblib.load('../../models/biyonix_nal_model.pkl')

dummy_data = pd.DataFrame({
    'Frequency (Hz)': [1000.0],
    'Age': [36.0],
    'Air Conduction': [55.0],
    'Bone Conduction': [50.0],
    'PTA': [60.0],
    'AC_250': [30.0], 'AC_500': [45.0], 'AC_750': [50.0], 'AC_1000': [55.0],
    'AC_1500': [60.0], 'AC_2000': [65.0], 'AC_3000': [70.0], 'AC_4000': [75.0],
    'AC_6000': [80.0], 'AC_8000': [85.0],
    'Ear Side': ['Right'],
    'Gender': ['Male']
})

for col in dummy_data.select_dtypes(include=['float64']).columns:
    dummy_data[col] = dummy_data[col].astype(np.float32)

print("Model ONNX formatına dönüştürülüyor, lütfen bekleyin...")

onnx_model = to_onnx(
    pipeline,
    dummy_data[:1],
    target_opset={'': 15, 'ai.onnx.ml': 3}
)

onnx_path = "../../models/biyonix_nal_model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Biyonix Modeli başarıyla '{onnx_path}' olarak kaydedildi!\n")
print("ONNX modeli test ediliyor (Inference)...")

sess = rt.InferenceSession(onnx_path)

inputs = {}
for col in dummy_data.columns:
    onnx_col_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)

    if col in ['Ear Side', 'Gender']:
        inputs[onnx_col_name] = dummy_data[col].values.astype(object).reshape(-1, 1)
    else:
        inputs[onnx_col_name] = dummy_data[col].values.astype(np.float32).reshape(-1, 1)

predictions = sess.run(None, inputs)
output_array = predictions[0][0]

print("-" * 40)
print("TEST SONUÇLARI (1000 Hz, Sağ Kulak):")
print(f"Target 50 : {output_array[0]:.2f} dB")
print(f"Target 65 : {output_array[1]:.2f} dB")
print(f"Target 80 : {output_array[2]:.2f} dB")
print(f"MPO       : {output_array[3]:.2f} dB")
print("-" * 40)
print("ONNX dosyası hatasız çalışıyor, C# tarafına gömülmeye hazır!")