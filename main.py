import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Membaca file seattle-weather.csv
iris = pd.read_csv('seattle-weather.csv')

# Melihat informasi dataset pada 5 baris pertama
print(iris.head())

# Melihat informasi dataset
print(iris.info())

# Menghilangkan kolom yang tidak penting
iris.drop('date', axis=1, inplace=True)

# Memisahkan atribut dan label
X = iris[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = iris['weather']

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi dataset menjadi data latih & data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=123)

# Membuat model Logistic Regression
log_model = LogisticRegression()

# Melakukan pelatihan model terhadap data
log_model.fit(X_train, y_train)

# Evaluasi Model
y_pred = log_model.predict(X_test)

# Menghitung akurasi (accuracy score)
acc_score = round(accuracy_score(y_pred, y_test), 3)
print('Accuracy: ', acc_score)

# Membuat DataFrame untuk membandingkan hasil prediksi dengan nilai aktual
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Tampilkan hasil prediksi
print(results.head(10))
