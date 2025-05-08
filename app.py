import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, mean_squared_error, 
                           r2_score, classification_report)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.multiclass import type_of_target

st.set_page_config(page_title="Data Analysis & Modeling App", layout="wide")
st.title("📊 Data Analysis & Machine Learning App")

st.sidebar.header("📂 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "json", "parquet", "html"],
    help="Upload CSV, Excel, JSON, Parquet, or HTML files"
)

if uploaded_file is None:
    st.session_state.df = None

if 'df' not in st.session_state:
    st.session_state.df = None
if 'encoded_cols' not in st.session_state:
    st.session_state.encoded_cols = []
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

def load_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith('.html'):
            return pd.read_html(uploaded_file.read())[0]
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

def identify_id_columns(df):
    id_cols = []
    for col in df.columns:
        if 'id' in col.lower() or 'ID' in col or 'Id' in col:
            if df[col].nunique() == len(df):
                id_cols.append(col)
    return id_cols

def determine_problem_type(y):
    target_type = type_of_target(y)
    if target_type == 'continuous':
        return 'regression'
    elif target_type in ['binary', 'multiclass']:
        return 'classification'
    else:
        return 'regression'

if uploaded_file is not None:
    st.session_state.df = load_file(uploaded_file)
    st.session_state.encoded_cols = []
    st.session_state.selected_features = []
    st.session_state.target_col = None

if st.session_state.df is not None:
    st.header("🔍 Data Preview")
    st.write("First 5 rows:")
    st.dataframe(st.session_state.df.head())
    
    st.write("Data Types:")
    st.write(st.session_state.df.dtypes)

    id_cols = identify_id_columns(st.session_state.df)
    if id_cols:
        st.warning(f"⚠️ ID columns detected: {', '.join(id_cols)}. These will be excluded from modeling.")

    st.header("🛠️ Missing Values Handling")
    if st.session_state.df.isnull().sum().sum() > 0:
        st.write("Missing values count:")
        st.write(st.session_state.df.isnull().sum())
        
        missing_option = st.selectbox(
            "How to handle missing values?",
            ["Drop rows with missing values", 
             "Fill with mean", 
             "Fill with median", 
             "Fill with mode"]
        )
        
        if st.button("Apply Missing Value Treatment"):
            if missing_option == "Drop rows with missing values":
                st.session_state.df = st.session_state.df.dropna()
            elif missing_option == "Fill with mean":
                st.session_state.df = st.session_state.df.fillna(st.session_state.df.mean(numeric_only=True))
            elif missing_option == "Fill with median":
                st.session_state.df = st.session_state.df.fillna(st.session_state.df.median(numeric_only=True))
            elif missing_option == "Fill with mode":
                st.session_state.df = st.session_state.df.fillna(st.session_state.df.mode().iloc[0])
            st.success("✅ Missing values treated successfully!")
            st.write("Updated missing values count:")
            st.write(st.session_state.df.isnull().sum())
    else:
        st.success("✅ No missing values found in the dataset!")

    st.header("🔢 Feature Encoding")
    object_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if object_cols:
        st.write("Categorical columns to encode:")
        cols_to_encode_options = [col for col in object_cols if col not in id_cols]
        cols_to_encode = st.multiselect("Select columns to encode", cols_to_encode_options)
        
        if cols_to_encode and st.button("Encode Selected Columns"):
            le = LabelEncoder()
            for col in cols_to_encode:
                try:
                    st.session_state.df[col] = le.fit_transform(st.session_state.df[col].astype(str))
                    st.session_state.encoded_cols.append(col)
                except Exception as e:
                    st.error(f"Error encoding column {col}: {str(e)}")
            st.success("✅ Encoding completed!")
            st.write("Encoded columns:", st.session_state.encoded_cols)
            st.write("Updated data types:")
            st.write(st.session_state.df.dtypes)
    else:
        st.info("ℹ️ No categorical columns found for encoding.")

    if st.session_state.encoded_cols:
        st.info(f"ℹ️ Encoded columns: {', '.join(st.session_state.encoded_cols)}")

    st.header("🤖 Model Training")
    all_columns = [col for col in st.session_state.df.columns.tolist() if col not in id_cols]
    st.session_state.target_col = st.selectbox("Select target variable (y)", all_columns)

    feature_cols = [col for col in all_columns if col != st.session_state.target_col]

    if feature_cols:
        st.session_state.selected_features = st.multiselect(
            "Select features (X)", 
            feature_cols, 
            default=feature_cols
        )

        st.info(f"ℹ️ Selected target (y): {st.session_state.target_col}")
        st.info(f"ℹ️ Selected features (X): {', '.join(st.session_state.selected_features) if st.session_state.selected_features else 'None'}")

        if st.session_state.selected_features and st.button("Train All Models"):
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_col]

            if y.dtype == 'object':
                try:
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                    st.info("ℹ️ Target variable was encoded for modeling")
                except Exception as e:
                    st.error(f"❌ Failed to encode target variable: {str(e)}")
                    st.stop()

            if pd.Series(y).isnull().sum() > 0:
                st.warning(f"⚠️ Target column has missing values. Dropping rows with missing target values.")
                valid_indices = ~pd.Series(y).isnull()
                X = X[valid_indices]
                y = y[valid_indices]

            if X.isnull().sum().sum() > 0:
                st.warning("⚠️ Features still contain missing values. Using automatic imputation.")

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns

            numeric_transformer = make_pipeline(
                SimpleImputer(strategy='mean'),
                StandardScaler()
            )

            categorical_transformer = make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(handle_unknown='ignore')
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            problem_type = determine_problem_type(y_train)
            st.info(f"ℹ️ Problem type automatically detected as: {problem_type}")

            if problem_type == "regression":
                st.subheader("📈 Regression Models Performance")
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree Regression": DecisionTreeRegressor(),
                    "KNN Regression": KNeighborsRegressor()
                }
                results = []
                for name, model in models.items():
                    try:
                        pipe = make_pipeline(preprocessor, model)
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results.append({"Model": name, "MSE": f"{mse:.4f}", "R² Score": f"{r2:.4f}"})
                    except Exception as e:
                        st.error(f"Error with {name}: {str(e)}")
                        results.append({"Model": name, "MSE": "Failed", "R² Score": "Failed"})
                results_df = pd.DataFrame(results)
                st.table(results_df.style.set_properties(**{
                    'background-color': '#f0f2f6',
                    'color': '#000000',
                    'border-color': 'white'
                }))
            else:
                st.subheader("📊 Classification Models Performance")
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "KNN Classifier": KNeighborsClassifier(),
                    "Decision Tree Classifier": DecisionTreeClassifier(),
                    "SVC (Linear Kernel)": SVC(kernel='linear', random_state=42)
                }
                results = []
                for name, model in models.items():
                    try:
                        pipe = make_pipeline(preprocessor, model)
                        pipe.fit(X_train, y_train)
                        y_pred = pipe.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        results.append({"Model": name, "Accuracy": f"{acc:.4f}", "Classification Report": classification_report(y_test, y_pred)})
                    except Exception as e:
                        st.error(f"Error with {name}: {str(e)}")
                        results.append({"Model": name, "Accuracy": "Failed", "Classification Report": f"Error: {str(e)}"})
                acc_df = pd.DataFrame([{k: v for k, v in res.items() if k != "Classification Report"} for res in results])
                st.table(acc_df.style.set_properties(**{
                    'background-color': '#f0f2f6',
                    'color': '#000000',
                    'border-color': 'white'
                }))
                for res in results:
                    with st.expander(f"See detailed report for {res['Model']}"):
                        st.text(res["Classification Report"])
        else:
            st.warning("⚠️ Please select at least one feature for modeling")
    else:
        st.warning("⚠️ No features available after selecting target variable")
else:
    st.info("ℹ️ Please upload a data file to get started")
