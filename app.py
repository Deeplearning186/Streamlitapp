import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    df = pd.get_dummies(df)
    return df

@st.cache_resource
def train_models(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    trained = {}
    reports = {}
    probs = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        reports[name] = (report, auc, y_pred)
        probs[name] = model.predict_proba(X_test_scaled)[:, 1]
        trained[name] = model
    return trained, reports, probs, X_test_scaled, y_test

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_roc_curves(probs, y_test):
    fig, ax = plt.subplots()
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True)
    return fig

def shap_plots(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    # Bar plot
    fig_bar, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    fig_bar = plt.gcf()
    # Waterfall plot for first instance
    fig_water, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    fig_water = plt.gcf()
    return fig_bar, fig_water

def main():
    st.title("Telco Customer Churn Dashboard")
    df = load_data()
    trained, reports, probs, X_test_scaled, y_test = train_models(df)
    page = st.sidebar.selectbox("Select Page", ["EDA", "Model Performance", "SHAP"])

    if page == "EDA":
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['MonthlyCharges'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Total Charges Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['TotalCharges'], ax=ax)
        st.pyplot(fig)

        st.subheader("Churn Count")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif page == "Model Performance":
        st.subheader("Classification Reports")
        for name, (report, auc, y_pred) in reports.items():
            st.markdown(f"### {name}")
            st.write(f"ROC AUC: {auc:.3f}")
            st.json(report)
            st.pyplot(plot_confusion_matrix(y_test, y_pred))

        st.subheader("ROC Curve Comparison")
        st.pyplot(plot_roc_curves(probs, y_test))

    else:
        st.subheader("SHAP Explainability (XGBoost)")
        fig_bar, fig_water = shap_plots(trained["XGBoost"], X_test_scaled)
        st.pyplot(fig_bar)
        st.pyplot(fig_water)

if __name__ == "__main__":
    main()
