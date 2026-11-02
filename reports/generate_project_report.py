"""Generate a PDF summary of the real estate advisor project."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "real_estate_investment_report.pdf"
FIGURES = ROOT / "reports" / "figures"
INK = colors.HexColor("#17343B")
TEAL = colors.HexColor("#117C78")
PALE = colors.HexColor("#EAF3F1")
MUTED = colors.HexColor("#52656A")
RULE = colors.HexColor("#D8E1E0")


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=32,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=11,
            leading=16,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=7,
        ),
        "subheading": ParagraphStyle(
            "Subheading",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=TEAL,
            spaceBefore=5,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontSize=9,
            leading=13,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "table_body": ParagraphStyle(
            "TableBody",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "center": ParagraphStyle(
            "MetricValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["BodyText"],
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


def p(text: str, style):
    return Paragraph(text, style)


def make_table(rows, widths, header=True):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE))
    table.setStyle(TableStyle(commands))
    return table


def chart(path: Path, width: float):
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        height = width * image.height / image.width
    return Image(str(path), width=width, height=height)


def page_chrome(canvas, document):
    canvas.saveState()
    page_width, page_height = A4
    canvas.setFillColor(INK)
    canvas.rect(0, page_height - 0.42 * inch, page_width, 0.42 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(0.62 * inch, page_height - 0.27 * inch, "REAL ESTATE INVESTMENT ADVISOR")
    canvas.setStrokeColor(RULE)
    canvas.line(0.62 * inch, 0.48 * inch, page_width - 0.62 * inch, 0.48 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(0.62 * inch, 0.30 * inch, "Project report | Synthetic-data demonstration")
    canvas.drawRightString(page_width - 0.62 * inch, 0.30 * inch, f"{document.page}")
    canvas.restoreState()


def build_report():
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.67 * inch,
        bottomMargin=0.62 * inch,
        title="Real Estate Investment Advisor - Project Report",
        author="Real Estate Investment Advisor Project",
    )
    story = []

    story.extend(
        [
            Spacer(1, 0.22 * inch),
            p("REAL ESTATE / DATA PROJECT", styles["subheading"]),
            p("Real Estate<br/>Investment Advisor", styles["title"]),
            p(
                "Project report: data quality, target design, exploratory findings, and deployed application",
                styles["subtitle"],
            ),
            p(f"Prepared {date.today().isoformat()}", styles["small"]),
            Spacer(1, 0.16 * inch),
            p("Executive summary", styles["heading"]),
            p(
                "This project demonstrates an end-to-end housing-listing workflow: clean and engineer listing data, "
                "explore its patterns, define illustrative investment targets, and serve results through Streamlit. "
                "The source data is synthetic-looking and prices are unusually similar across locations. "
                "Treat the app as a portfolio and teaching demonstration, not as Indian market research or investment advice.",
                styles["body"],
            ),
        ]
    )

    metrics = [
        [p("250,000", styles["center"]), p("20", styles["center"]), p("42", styles["center"]), p("500", styles["center"])],
        [p("listings", styles["metric_label"]), p("states", styles["metric_label"]), p("cities", styles["metric_label"]), p("localities", styles["metric_label"])],
    ]
    metric_table = Table(metrics, colWidths=[1.72 * inch] * 4, hAlign="LEFT")
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([metric_table, Spacer(1, 0.16 * inch), p("Project deliverables", styles["subheading"])])
    story.extend(
        [
            p("- Streamlit pages: Predict, Explore, and Insights.", styles["body"]),
            p("- Processed dataset, EDA charts, preprocessing report, and methodology notes.", styles["body"]),
            p("- Public source repository and hosted Streamlit application.", styles["body"]),
            p("- Rule-based prediction fallback; trained model pipelines and benchmark metrics are not currently available.", styles["body"]),
            PageBreak(),
        ]
    )

    story.extend([p("1. Data and preparation", styles["heading"])])
    story.append(
        p(
            "The source table contains 250,000 records and 23 columns, with no missing values or duplicate rows. "
            "The analysis covers 20 states, 42 cities, and 500 localities.",
            styles["body"],
        )
    )
    quality_rows = [
        [p("Check", styles["table_header"]), p("Finding", styles["table_header"]), p("Treatment", styles["table_header"])],
        [p("Price per square foot", styles["table_body"]), p("Original field is rounded and in lakhs/sq ft.", styles["table_body"]), p("Recomputed in INR as price in lakhs x 100,000 / area.", styles["table_body"])],
        [p("Floor values", styles["table_body"]), p("46.52% have floor number above total floors.", styles["table_body"]), p("Kept; flagged and capped only for floor ratio.", styles["table_body"])],
        [p("INR/sq ft outliers", styles["table_body"]), p("19,723 records, or 7.89%, flagged by IQR.", styles["table_body"]), p("Kept in the data; flags retained for analysis.", styles["table_body"])],
        [p("Property age", styles["table_body"]), p("Matches 2025 minus year built.", styles["table_body"]), p("Recomputed age and mismatch flag stored.", styles["table_body"])],
    ]
    story.extend([make_table(quality_rows, [1.25 * inch, 2.65 * inch, 2.95 * inch]), Spacer(1, 8)])
    story.append(
        p(
            "An infrastructure proxy combines nearby schools, nearby hospitals, and ordinal public-transport access. "
            "It is not an official infrastructure index. No crime score or listing date is present in the source.",
            styles["body"],
        )
    )
    charts_row = Table(
        [[chart(FIGURES / "q01_price_distribution.png", 3.05 * inch), chart(FIGURES / "q04_size_vs_price.png", 3.05 * inch)]],
        colWidths=[3.25 * inch, 3.25 * inch],
        hAlign="LEFT",
    )
    charts_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story.extend(
        [
            charts_row,
            p("Price distribution and size-price relationship in the source table.", styles["small"]),
            PageBreak(),
        ]
    )

    story.extend([p("2. Targets and modelling", styles["heading"])])
    story.append(
        p(
            "The project defines targets from explicit rules so the demo is reproducible. They are not independent evidence of real-world investment performance.",
            styles["body"],
        )
    )
    target_rows = [
        [p("Investment score point", styles["table_header"]), p("Rule", styles["table_header"])],
        [p("1", styles["table_body"]), p("INR/sq ft at or below the city median.", styles["table_body"])],
        [p("2", styles["table_body"]), p("At least 3 bedrooms (BHK).", styles["table_body"])],
        [p("3", styles["table_body"]), p("Ready to move.", styles["table_body"])],
        [p("4", styles["table_body"]), p("Infrastructure proxy above the dataset median.", styles["table_body"])],
        [p("5", styles["table_body"]), p("Amenity count above the dataset median.", styles["table_body"])],
        [p("6", styles["table_body"]), p("High public-transport access.", styles["table_body"])],
    ]
    story.extend([make_table(target_rows, [1.35 * inch, 5.5 * inch]), Spacer(1, 8)])
    story.append(
        p(
            "Good Investment is labelled positive at score >= 3. The dataset has 146,530 positive and 103,470 negative records. "
            "The label is rule-derived, so high classification scores would not mean that a model discovered an independently validated investment signal.",
            styles["body"],
        )
    )
    story.append(p("Illustrative five-year value", styles["subheading"]))
    story.append(
        p(
            "Future price = current price x (1 + annual rate)^5. The annual rate combines a city-tier base (6%, 8%, or 10%), "
            "a property-type adjustment (0 to 1.5 percentage points), and up to 2 percentage points from the infrastructure proxy; "
            "the final rate is clipped to 5%-13%. The mean target rate in the processed dataset is 9.78%.",
            styles["body"],
        )
    )
    story.append(
        p(
            "Model status: the published app currently uses the documented rule-based fallback because trained classifier/regressor "
            "pipelines and a completed model-comparison report are absent. Do not interpret fallback outputs as trained-model predictions.",
            styles["body"],
        )
    )
    story.extend([PageBreak(), p("3. Exploratory findings and use", styles["heading"])])
    story.append(
        p(
            "The strongest conclusion is a limitation of the source data: prices and price per square foot barely vary by city or state. "
            "City mean prices range only about 250.8-258.5 lakhs, and state mean price per square foot ranges about INR 12,931-13,252. "
            "Amenities, parking, furnishing, facing, nearby schools, and hospitals show little price separation.",
            styles["body"],
        )
    )
    story.append(
        p(
            "The good-investment rate rises with high public transport partly by construction, because transport is one of the six label rules. "
            "This is a target-design relationship, not causal evidence that transit access will raise returns in a real market.",
            styles["body"],
        )
    )
    insight_charts = Table(
        [[chart(FIGURES / "q06_avg_pps_by_state.png", 3.05 * inch), chart(FIGURES / "q20_transport_pps_good_investment.png", 3.05 * inch)]],
        colWidths=[3.25 * inch, 3.25 * inch],
        hAlign="LEFT",
    )
    insight_charts.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([insight_charts, p("State-level price similarity and transport-related target rate.", styles["small"])])
    story.append(p("Application and intended use", styles["subheading"]))
    story.append(
        p(
            "Predict screens a hypothetical listing and estimates value under the project rate table. Explore filters and compares listings. "
            "Insights presents EDA charts and the data caveats. Use the app for portfolio demonstration and data-workflow education only; "
            "it is not a valuation, purchase recommendation, or substitute for local comparable sales, title checks, and site inspection.",
            styles["body"],
        )
    )
    links = [
        [p("Live Streamlit app", styles["table_body"]), p('<link href="https://real-estate-investment-advisor-b7ccphj93ddxkcdt5r8ybg.streamlit.app/" color="#117C78">Open deployed app</link>', styles["table_body"])],
        [p("Public GitHub repository", styles["table_body"]), p('<link href="https://github.com/priyanka-solanki07/real-estate-investment-advisor" color="#117C78">View source code</link>', styles["table_body"])],
    ]
    story.extend([make_table(links, [2.1 * inch, 4.75 * inch], header=False), Spacer(1, 8)])
    story.append(
        p(
            "Sources: project preprocessing report, methodology, findings, and EDA charts. Figures describe the supplied synthetic dataset only.",
            styles["small"],
        )
    )

    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())