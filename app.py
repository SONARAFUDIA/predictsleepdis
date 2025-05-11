import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset untuk training model
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Normal').astype(object)

# Preprocessing
cat_features = ['Gender', 'Occupation', 'BMI Category']
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(df[cat_features])
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(cat_features))

num_features = ['Age', 'Sleep Duration']
scaler = StandardScaler()
scaled_num = scaler.fit_transform(df[num_features])
scaled_df = pd.DataFrame(scaled_num, columns=num_features)

final_df = pd.concat([scaled_df, encoded_df, df['Sleep Disorder']], axis=1)

X = final_df.drop('Sleep Disorder', axis=1)
y = final_df['Sleep Disorder'].apply(lambda x: 0 if x == 'Normal' else 1)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Streamlit App
st.title("Prediksi Gangguan Tidur dan Rekomendasi Kesehatan")

st.header("Masukkan Data Anda")

age = st.slider("Usia", 18, 100, 25)
sleep_duration = st.slider("Durasi Tidur per Hari (jam)", 0.0, 12.0, 7.0)
gender = st.selectbox("Jenis Kelamin", ['Male', 'Female'])
occupation = st.selectbox("Pekerjaan", df['Occupation'].unique())
bmi_category = st.selectbox("Kategori BMI", ['Normal', 'Overweight', 'Obese'])

# Preprocessing input user
input_df = pd.DataFrame({
    'Age': [age],
    'Sleep Duration': [sleep_duration],
    'Gender': [gender],
    'Occupation': [occupation],
    'BMI Category': [bmi_category]
})

# Encode input
encoded_input = encoder.transform(input_df[cat_features])
encoded_input_df = pd.DataFrame(encoded_input.toarray(), columns=encoder.get_feature_names_out(cat_features))

# Scale numeric
scaled_input = scaler.transform(input_df[num_features])
scaled_input_df = pd.DataFrame(scaled_input, columns=num_features)

# Final input
user_data = pd.concat([scaled_input_df, encoded_input_df], axis=1)

# Ensure same columns as training data
for col in X.columns:
    if col not in user_data.columns:
        user_data[col] = 0  # Add missing columns with 0

user_data = user_data[X.columns]  # Reorder to match training data

# Predict
prediction = rf.predict(user_data)[0]

st.subheader("Hasil Prediksi:")
if prediction == 1:
    st.error("Anda Mungkin Mengalami Gangguan Tidur!")
    # st.markdown("""
    # ### Rekomendasi:
    # - Konsultasikan dengan dokter atau ahli tidur.
    # - Terapkan rutinitas tidur yang konsisten.
    # - Hindari penggunaan gawai 1 jam sebelum tidur.
    # - Kurangi konsumsi kafein dan alkohol.
    # - Perhatikan kondisi kamar tidur: gelap, sejuk, dan tenang.
    # """)
else:
    st.success("Tidur Anda Normal.")
    # st.markdown("""
    # ### Tips Menjaga Kualitas Tidur:
    # - Pertahankan jadwal tidur yang konsisten.
    # - Olahraga secara teratur.
    # - Hindari makan berat sebelum tidur.
    # - Ciptakan suasana tidur yang nyaman.
    # """)

