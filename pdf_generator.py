import os
import datetime
from fpdf import FPDF
from PIL import Image
import qrcode

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, "Eye Disease Detection Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        
def generate_pdf(name, age, image, predictions, lang="English", gradcam_image=None,
                 cm_image=None):
    """
    Generates a PDF medical report for the eye disease detection results.
    
    Args:
        name (str): Patient name
        age (str/int): Patient age
        image (PIL.Image): Original eye image
        predictions (list of dict): [{"disease": str, "confidence": float}]
        lang (str): "English" or "Hindi"
        gradcam_image (PIL.Image, optional): Grad-CAM heatmap overlay
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf = PDFReport()
    pdf.add_page()

    # Patient Info
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Date & Time: {timestamp}", ln=True)
    pdf.cell(0, 10, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.ln(5)

    # Add Original Image
    img_path = "temp_eye_image.jpg"
    image.save(img_path)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Original Eye Image:", ln=True)
    pdf.image(img_path, w=80)
    os.remove(img_path)

    # Grad-CAM Image
    if gradcam_image:
        grad_path = "temp_gradcam_image.jpg"
        gradcam_image.save(grad_path)
        pdf.ln(5)
        pdf.cell(0, 10, "Grad-CAM Heatmap:", ln=True)
        pdf.image(grad_path, w=80)
        os.remove(grad_path)

    # Predictions Table
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Prediction Results:", ln=True)
    pdf.set_font("Helvetica", size=12)
    for pred in predictions:
        disease = pred["disease"]
        conf = round(pred["confidence"] * 100, 2)
        pdf.cell(0, 10, f"Disease: {disease}  |  Confidence: {conf}%", ln=True)

    # QR Code (optional)
    qr_data = f"Patient: {name}\nDisease: {predictions[0]['disease']}\nConfidence: {predictions[0]['confidence']*100:.2f}%"
    qr = qrcode.make(qr_data)
    qr_path = "temp_qr.png"
    qr.save(qr_path)
    pdf.ln(5)
    pdf.cell(0, 10, "QR Code:", ln=True)
    pdf.image(qr_path, w=40)
    os.remove(qr_path)
     # Confusion Matrix
    if cm_image and os.path.exists(cm_image):
        pdf.ln(5)
        pdf.cell(0, 10, "Confusion Matrix:", ln=True)
        pdf.image(cm_image, w=100)

    # Save final PDF
    pdf_path = f"report_{name.replace(' ', '_')}.pdf"
    pdf.output(pdf_path)
    return pdf_path
