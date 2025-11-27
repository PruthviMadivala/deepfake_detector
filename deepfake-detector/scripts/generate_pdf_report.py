from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os

REPORT_FILE = os.path.abspath(os.path.join("..", "downloads", "deepfake_report.pdf"))

VIDEO_NAME = "trump.mp4"
PREDICTION = "REAL"
CONFIDENCE = 0.12
AUDIO_SYNC = 1  # 1 = good, 0.5 = medium, 0 = poor

FACE_IMG = "gradcam_face.jpg"
HEATMAP_IMG = "gradcam_heatmap.jpg"
OVERLAY_IMG = "gradcam_overlay.jpg"


# ---------------------------------------------------------
# SAFE IMAGE LOADER
# ---------------------------------------------------------
def add_image(path, width=450):
    if not os.path.exists(path):
        return Paragraph(f"<font color='red'>⚠ Image not found: {path}</font>", getSampleStyleSheet()['BodyText'])

    return Image(path, width=width, height=width)


# ---------------------------------------------------------
# BUILD REPORT
# ---------------------------------------------------------
def build_report():
    print("📄 Generating upgraded PDF report...")

    doc = SimpleDocTemplate(
        REPORT_FILE,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="Title",
        fontSize=24,
        leading=28,
        alignment=1,
        textColor=colors.HexColor("#2b2b2b")
    )
    heading = ParagraphStyle(
        name="Heading",
        fontSize=18,
        leading=22,
        spaceAfter=10,
        textColor=colors.HexColor("#333333")
    )
    body = styles["BodyText"]

    elements = []

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    elements.append(Paragraph("Deepfake Analysis Report", title_style))
    elements.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # Summary Table
    # ---------------------------------------------------------
    summary_data = [
        ["Parameter", "Value"],
        ["Video File", VIDEO_NAME],
        ["Prediction", PREDICTION],
        ["Confidence Score", f"{CONFIDENCE:.4f}"],
        ["Audio-Video Sync Score", AUDIO_SYNC],
    ]

    summary_table = Table(summary_data, colWidths=[160, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#ececec")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#000000")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # Full Face Image
    # ---------------------------------------------------------
    elements.append(Paragraph("Detected Face (High Resolution)", heading))
    elements.append(add_image(FACE_IMG, width=350))
    elements.append(Spacer(1, 25))

    # ---------------------------------------------------------
    # Grad-CAM Images: Heatmap + Overlay Side-by-Side
    # ---------------------------------------------------------
    elements.append(Paragraph("Grad-CAM Visualization", heading))

    heatmap = add_image(HEATMAP_IMG, width=250)
    overlay = add_image(OVERLAY_IMG, width=250)

    img_table = Table([[heatmap, overlay]], colWidths=[260, 260])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))

    elements.append(img_table)
    elements.append(Spacer(1, 20))

    # ---------------------------------------------------------
    # Description
    # ---------------------------------------------------------
    elements.append(Paragraph(
        "The Grad-CAM heatmap highlights the areas influencing the model's decision. "
        "Red regions indicate zones with stronger importance during classification.",
        body
    ))

    elements.append(Spacer(1, 30))

    # ---------------------------------------------------------
    # Footer
    # ---------------------------------------------------------
    elements.append(Paragraph(
        "<i>Generated automatically by Deepfake Detector System</i>",
        ParagraphStyle(name="Footer", alignment=1, fontSize=10)
    ))

    # Save PDF
    doc.build(elements)
    print(f"✅ Report generated: {REPORT_FILE}")


if __name__ == "__main__":
    build_report()
