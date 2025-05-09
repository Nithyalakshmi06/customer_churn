import io
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# Function to generate a chart
def generate_churn_chart():
    labels = ['Churned', 'Not Churned']
    sizes = [40, 60]  # Example data
    colors = ['red', 'green']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close(fig)

    return img_buffer

# Function to generate a PDF report
def generate_pdf_report():
    pdf_buffer = io.BytesIO()

    # Create the PDF document with the buffer
    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    story = [
        Paragraph("Customer Churn Prediction Report", styles['Title']),
        Spacer(1, 12),
        Paragraph("Churn Rate: 40%", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Retention Rate: 60%", styles['Normal']),
        Spacer(1, 12),
    ]

    # Generate churn chart and add it to the PDF
    churn_chart_buffer = generate_churn_chart()
    story.append(Image(churn_chart_buffer, width=400, height=300))

    # Build the PDF (No filename argument here)
    doc.build(story)

    pdf_buffer.seek(0)
    return pdf_buffer

# Streamlit code to display and download the PDF
pdf_buffer = generate_pdf_report()

st.download_button(
    label="Download PDF Report",
    data=pdf_buffer,
    file_name="churn_report.pdf",
    mime="application/pdf"
)

st.markdown(
    f'<a href="data:application/pdf;base64,{pdf_buffer.getvalue().encode("base64").decode()}" download="churn_report.pdf">Click here to view the PDF</a>',
    unsafe_allow_html=True
)
