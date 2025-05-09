import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Function to handle login page
def login_page():
    st.title('Login Page')
    st.write("Please enter your credentials to access the app.")
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success('Login successful!')
        else:
            st.error('Invalid credentials. Please try again.')

# Page 2: Dataset Prediction Results
def dataset_results_page():
    df = pd.read_csv('data/churn.csv')
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    df.columns = df.columns.str.strip()

    original_columns = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in original_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.title('📈 Customer Churn Prediction Dashboard')

    st.subheader('🔎 Key Insights')
    total_customers = len(df)
    churned_customers = sum(df['Churn'] == 1)
    churn_rate = churned_customers / total_customers * 100
    st.write(f"Total Customers: {total_customers}")
    st.write(f"Churned Customers: {churned_customers}")
    st.write(f"Churn Rate: {churn_rate:.2f}%")
    st.write(f"Model Accuracy: {accuracy:.2f}")

    st.subheader("📊 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
    st.pyplot(fig)

    st.subheader('🥧 Churn Distribution')
    churn_counts = df['Churn'].value_counts()
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(churn_counts, labels=['Stay', 'Churn'], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    st.subheader('📊 Churn by Contract Type')
    contract_churn = df.groupby('Contract')['Churn'].mean() * 100
    fig_bar, ax_bar = plt.subplots()
    contract_churn.plot(kind='bar', ax=ax_bar, color='purple')
    ax_bar.set_title('Churn Rate by Contract Type')
    ax_bar.set_xlabel('Contract Type')
    ax_bar.set_ylabel('Churn Rate (%)')
    st.pyplot(fig_bar)

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'], ax=ax_cm)
    ax_cm.set_ylabel('Actual')
    ax_cm.set_xlabel('Predicted')
    st.pyplot(fig_cm)

# Page 3: User Input Prediction Results
def input_prediction_page():
    df = pd.read_csv('data/churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    original_columns = df.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in original_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    model = RandomForestClassifier()
    model.fit(X, y)

    st.subheader("📥 Enter Customer Data for Prediction")
    input_data = {}
    for col in X.columns:
        if col in label_encoders:
            input_data[col] = st.selectbox(col, label_encoders[col].classes_)
        else:
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            default_val = int(df[col].mean())
            input_data[col] = st.slider(col, min_val, max_val, default_val)

    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    churn_prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.write('🔴 This customer is predicted to churn.')
        st.write(f'Churn Probability: {churn_prob * 100:.2f}%')
    else:
        st.write('🟢 This customer is predicted to stay.')
        st.write(f'Stay Probability: {(1 - churn_prob) * 100:.2f}%')

    # Pie Chart for Input Prediction
    st.subheader('📊 Prediction Breakdown')
    fig_pred, ax_pred = plt.subplots()
    ax_pred.pie(
        [1 - churn_prob, churn_prob],
        labels=['Stay', 'Churn'],
        colors=['green', 'red'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax_pred.axis('equal')
    st.pyplot(fig_pred)

    def generate_pdf(input_data, prediction, churn_prob):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Customer Churn Prediction Report")
        c.setFont("Helvetica", 12)
        c.drawString(100, 730, "Customer Information and Prediction Summary")
        y_position = 700
        for key, value in input_data.items():
            c.drawString(100, y_position, f"{key}: {value}")
            y_position -= 20
        c.drawString(100, y_position, f"Predicted Churn: {'Yes' if prediction == 1 else 'No'}")
        c.drawString(100, y_position - 20, f"Churn Probability: {churn_prob * 100:.2f}%")
        c.save()
        buffer.seek(0)
        return buffer

    if st.button('Generate Prediction Report'):
        pdf_buffer = generate_pdf(input_data, prediction, churn_prob)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="churn_prediction_report.pdf",
            mime="application/pdf"
        )

    st.subheader("📂 Bulk Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload customer CSV file", type=['csv'])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        for col in uploaded_df.columns:
            if col in label_encoders:
                uploaded_df[col] = label_encoders[col].transform(uploaded_df[col].astype(str))
        bulk_preds = model.predict(uploaded_df)
        uploaded_df['Prediction'] = ['Churn' if p == 1 else 'Stay' for p in bulk_preds]
        uploaded_df['Churn Probability (%)'] = model.predict_proba(uploaded_df)[:, 1] * 100
        st.dataframe(uploaded_df)
        st.download_button(
            label="Download Prediction Results",
            data=uploaded_df.to_csv(index=False).encode('utf-8'),
            file_name="bulk_churn_predictions.csv",
            mime='text/csv'
        )

# Main function to handle page navigation
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        login_page()
    else:
        page = st.sidebar.selectbox("Choose a page", ["Dataset Results", "Input Prediction"])
        if page == "Dataset Results":
            dataset_results_page()
        elif page == "Input Prediction":
            input_prediction_page()

if __name__ == '__main__':
    main()
