# pdf_report.py
from io import BytesIO
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def generate_pdf(prediction, confidence, probabilities, labels, image, case_id=None, patient_name=None, validation_info=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = []

    # Title
    content.append(Paragraph("Brain Tumor MRI Classification Report", styles["Title"]))
    content.append(Spacer(1, 10))

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content.append(Paragraph(f"<b>Date:</b> {timestamp}", styles["Normal"]))
    
    # Case ID & Patient Name
    if patient_name:
        content.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    if case_id:
        content.append(Paragraph(f"<b>Case ID:</b> {case_id}", styles["Normal"]))
        
    if validation_info:
        content.append(Paragraph(f"<b>MRI Validation:</b> {validation_info}", styles["Normal"]))

    content.append(Spacer(1, 10))

    # Prediction details
    content.append(Paragraph(f"<b>Tumor Classification:</b> {prediction}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Normal"]))
    content.append(Spacer(1, 10))

    # MRI Image
    if image is not None:
        # Save image to a temporary buffer to embed in PDF
        img_buffer = BytesIO()
        image.copy().convert("RGB").save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        
        # Calculate aspect ratio to fit the image
        img_width, img_height = image.size
        aspect = img_height / float(img_width)
        display_width = 3.5 * inch
        display_height = display_width * aspect
        
        rl_img = RLImage(img_buffer, width=display_width, height=display_height)
        content.append(rl_img)
        content.append(Spacer(1, 10))

    # Probabilities
    content.append(Paragraph("Class Probabilities", styles["Heading2"]))
    for label, prob in zip(labels, probabilities):
         content.append(Paragraph(f"{label}: {prob:.4f}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer
