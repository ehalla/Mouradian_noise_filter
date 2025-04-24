import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model

st.title("🫁 Plethysmography Noise Filter 🫁")

# Upload widgets
model_file = st.file_uploader("Upload Model (.keras)", type="keras")
scaler_file = st.file_uploader("Upload Scaler (.pkl)", type="pkl")
features_file = st.file_uploader("Upload Feature Columns (.pkl)", type="pkl")
data_file = st.file_uploader("Upload CSV Data File", type="csv")

if st.button("🔍 Run Model"):

    if not (model_file and scaler_file and features_file and data_file):
        st.warning("Please upload all required files.")
    else:
        try:
            # Save uploaded files
            with open("temp_model.keras", "wb") as f:
                f.write(model_file.getbuffer())
            with open("temp_scaler.pkl", "wb") as f:
                f.write(scaler_file.getbuffer())
            with open("temp_features.pkl", "wb") as f:
                f.write(features_file.getbuffer())

            # Load them AFTER saving
            model = load_model("temp_model.keras")
            scaler = joblib.load("temp_scaler.pkl")
            feature_columns = joblib.load("temp_features.pkl")

            # Load and clean CSV data
            df = pd.read_csv(data_file, encoding='ISO-8859-1')
            df = df.drop(df.index[0:2]).reset_index(drop=True)
            mean_idx = df[df['Raw-Pleth#1'] == 'Mean'].index
            drop_idx = set(mean_idx) | set(mean_idx + 1)
            df = df.drop(drop_idx, errors='ignore').reset_index(drop=True)
            df = df.replace({'−': '-'}, regex=True)
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # ✅ NEW: Convert 14th column to timedelta and seconds
            try:
                time_col_name = df.columns[13]
                df[time_col_name] = pd.to_timedelta(df[time_col_name].astype(str), errors='coerce')
                df['Time_seconds'] = df[time_col_name].dt.total_seconds()
            except Exception as e:
                st.warning(f"⚠️ Could not convert 14th column to time: {e}")

            df = df.applymap(lambda x: x if str(x).replace('.', '', 1).replace('-', '', 1).isdigit() else None)
            df = df.apply(pd.to_numeric, errors='coerce')

            # Filter and predict
            df_clean = df.select_dtypes(include=[float, int]).dropna(axis=1, how='all')
            for col in feature_columns:
                if col not in df_clean.columns:
                    df_clean[col] = 0
            df_clean = df_clean[feature_columns]
            X_scaled = scaler.transform(df_clean)
            y_pred = (model.predict(X_scaled) > 0.5).astype(int).flatten()
            df['deleted_flag'] = y_pred
            kept_df = df[df['deleted_flag'] == 0]

            # Show and download
            st.success(f"✅ Your data has been successfully filtered!")
            st.dataframe(kept_df.head())

            csv = kept_df.to_csv(index=False).encode('utf-8')
            st.download_button("📁 Download Cleaned CSV", data=csv, file_name="cleaned_output.csv")

        except Exception as e:
            st.error(f"❌ Something went wrong:\n\n{e}")
