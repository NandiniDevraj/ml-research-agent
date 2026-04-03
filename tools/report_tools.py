# tools/report_tools.py
import os
import json
import mlflow
import pandas as pd
from datetime import datetime
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()


def setup_mlflow():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )


@tool
def get_experiment_results() -> str:
    """
    Fetch all experiment results from MLflow for the report.
    Returns model names, metrics and which one was registered.
    """
    try:
        setup_mlflow()
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name("hospital-readmission")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.auc DESC"]
        )

        results = []
        for run in runs:
            results.append({
                "model":     run.info.run_name,
                "auc":       run.data.metrics.get("auc", 0),
                "f1":        run.data.metrics.get("f1", 0),
                "precision": run.data.metrics.get("precision", 0),
                "recall":    run.data.metrics.get("recall", 0),
                "accuracy":  run.data.metrics.get("accuracy", 0),
            })

        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_dataset_stats(file_path: str) -> str:
    """
    Get final dataset statistics for the report.
    """
    try:
        df = pd.read_csv(file_path)
        return json.dumps({
            "rows":             len(df),
            "columns":          len(df.columns),
            "features":         df.columns.tolist(),
            "class_balance":    df["readmitted"].value_counts().to_dict(),
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def save_report_as_markdown(report_content: str) -> str:
    """
    Save the research report as a markdown file locally.
    Returns the file path.
    """
    try:
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"reports/ml_research_report_{timestamp}.md"

        with open(path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return json.dumps({
            "status": "saved",
            "path": path
        })
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def convert_markdown_to_pdf(markdown_path: str) -> str:
    """
    Convert the markdown report to a styled PDF file.
    Returns the PDF file path.
    """
    try:
        import re

        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        pdf_path = markdown_path.replace(".md", ".pdf")

        # Convert markdown to styled HTML then to PDF
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        h1_style = ParagraphStyle(
            "CustomH1",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#16213e"),
            spaceBefore=16,
            spaceAfter=8,
            borderPad=4
        )
        h2_style = ParagraphStyle(
            "CustomH2",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#0f3460"),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6
        )
        bullet_style = ParagraphStyle(
            "CustomBullet",
            parent=styles["Normal"],
            fontSize=10,
            leading=16,
            leftIndent=20,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4
        )

        story = []
        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                story.append(Spacer(1, 6))
                i += 1
                continue

            # Title
            if line.startswith("# "):
                text = line[2:].strip()
                story.append(Paragraph(text, title_style))
                story.append(Spacer(1, 12))

            # H2
            elif line.startswith("## "):
                text = line[3:].strip()
                story.append(Paragraph(text, h1_style))

            # H3
            elif line.startswith("### "):
                text = line[4:].strip()
                story.append(Paragraph(text, h2_style))

            # Table rows
            elif line.startswith("|"):
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row_line = lines[i].strip()
                    if not re.match(r"^\|[-| :]+\|$", row_line):
                        cells = [
                            c.strip()
                            for c in row_line.strip("|").split("|")
                        ]
                        table_lines.append(cells)
                    i += 1

                if table_lines:
                    col_count = len(table_lines[0])
                    col_width  = (6.5 * inch) / col_count
                    col_widths = [col_width] * col_count

                    table = Table(table_lines, colWidths=col_widths)
                    table.setStyle(TableStyle([
                        ("BACKGROUND",  (0, 0), (-1, 0),
                         colors.HexColor("#16213e")),
                        ("TEXTCOLOR",   (0, 0), (-1, 0),
                         colors.white),
                        ("FONTNAME",    (0, 0), (-1, 0),
                         "Helvetica-Bold"),
                        ("FONTSIZE",    (0, 0), (-1, 0), 10),
                        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                         [colors.white, colors.HexColor("#f0f4f8")]),
                        ("GRID",        (0, 0), (-1, -1),
                         0.5, colors.HexColor("#cccccc")),
                        ("FONTSIZE",    (0, 1), (-1, -1), 9),
                        ("PADDING",     (0, 0), (-1, -1), 6),
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 12))
                continue

            # Bullet points
            elif line.startswith("- ") or line.startswith("* "):
                text = "• " + line[2:].strip()
                text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
                story.append(Paragraph(text, bullet_style))

            # Numbered list
            elif re.match(r"^\d+\.", line):
                text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
                story.append(Paragraph(text, bullet_style))

            # Regular paragraph
            else:
                text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
                text = re.sub(r"`(.*?)`", r"<i>\1</i>", text)
                story.append(Paragraph(text, body_style))

            i += 1

        doc.build(story)

        return json.dumps({
            "status": "success",
            "pdf_path": pdf_path
        })
    except Exception as e:
        return f"Error converting to PDF: {str(e)}"