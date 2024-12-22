import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Membaca file seattle-weather.csv
iris = pd.read_csv('seattle-weather.csv')

# Melihat informasi dataset pada 5 baris pertama
print(iris.head())

# Melihat informasi dataset
print(iris.info())

# Menghilangkan kolom yang tidak penting
iris.drop('date', axis=1, inplace=True)

# Menggunakan LabelEncoder untuk mengonversi kategori menjadi angka
label_encoder = LabelEncoder()
iris['weather'] = label_encoder.fit_transform(iris['weather'])

# Memisahkan atribut dan label
X = iris[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = iris['weather']

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi dataset menjadi data latih & data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=123)

# Membuat model Linear Regression
lin_model = LinearRegression()

# Melakukan pelatihan model terhadap data
lin_model.fit(X_train, y_train)

# Evaluasi Model
y_pred = lin_model.predict(X_test)

# Mengubah prediksi kontinu menjadi kelas dengan pembulatan
y_pred_classes = [round(x) for x in y_pred]

# Menghitung akurasi
acc_score = round(accuracy_score(y_test, y_pred_classes), 3)
print('Accuracy: ', acc_score)

# Membalikkan encoding untuk hasil prediksi dan nilai aktual
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Membuat DataFrame untuk membandingkan hasil prediksi dengan nilai aktual
results = pd.DataFrame({
    'Actual': y_test_labels,
    'Predicted': y_pred_labels
})

# Tampilkan hasil prediksi
print(results.head(50))
