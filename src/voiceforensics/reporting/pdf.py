"""Court-oriented PDF report generation (ReportLab).

Renders an :class:`AnalysisResult` (plus optional dense artifacts) into a
chain-of-custody report suitable as a forensic exhibit. The report is explicit
about its own limitations and about model provenance (heuristic baseline vs.
trained backends).
"""

from __future__ import annotations

import datetime as _dt
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from voiceforensics import __version__
from voiceforensics.reporting import templates
from voiceforensics.schemas import AnalysisResult, Verdict

if TYPE_CHECKING:
    from voiceforensics.pipeline import AnalysisArtifacts

_VERDICT_COLOR = {
    Verdict.AUTHENTIC: colors.HexColor("#1a7f37"),
    Verdict.LEANING_AUTHENTIC: colors.HexColor("#4f8f2f"),
    Verdict.UNCERTAIN: colors.HexColor("#9a6700"),
    Verdict.LIKELY_SYNTHETIC: colors.HexColor("#bc4c00"),
    Verdict.HIGH_CONFIDENCE_SYNTHETIC: colors.HexColor("#cf222e"),
}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("VFBody", parent=ss["BodyText"], fontSize=9.5, leading=14))
    ss.add(ParagraphStyle("VFSmall", parent=ss["BodyText"], fontSize=8, leading=11,
                          textColor=colors.HexColor("#57606a")))
    ss.add(ParagraphStyle("VFH1", parent=ss["Heading1"], fontSize=15, spaceAfter=6))
    ss.add(ParagraphStyle("VFH2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("VFTitle", parent=ss["Title"], fontSize=22, leading=26))
    return ss


def _kv_table(rows: list[tuple[str, str]], styles) -> Table:
    data = [[Paragraph(f"<b>{k}</b>", styles["VFBody"]), Paragraph(v, styles["VFBody"])] for k, v in rows]
    t = Table(data, colWidths=[5.0 * cm, 11.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#d0d7de")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def _footer_factory(file_hash: str, baseline_only: bool):
    short = file_hash[:16]

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#57606a"))
        tag = "  |  HEURISTIC BASELINE" if baseline_only else ""
        canvas.drawString(2 * cm, 1.1 * cm, f"VoiceForensics report  |  SHA-256:{short}\u2026{tag}")
        canvas.drawRightString(19 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#d0d7de"))
        canvas.line(2 * cm, 1.4 * cm, 19 * cm, 1.4 * cm)
        canvas.restoreState()

    return _footer


def _render_exhibits(artifacts: AnalysisArtifacts | None, result: AnalysisResult, workdir: Path) -> list[tuple[str, Path]]:
    if artifacts is None:
        return []
    from voiceforensics.viz.render import heatmap_png, mel_spectrogram_png, waveform_png

    exhibits: list[tuple[str, Path]] = []
    exhibits.append(
        ("Exhibit A \u2014 Mel spectrogram",
         mel_spectrogram_png(artifacts.mel, artifacts.sample_rate, workdir / "mel.png"))
    )
    exhibits.append(
        ("Exhibit B \u2014 Synthetic-artifact heatmap",
         heatmap_png(artifacts.heatmap, result.result.segments, workdir / "heat.png"))
    )
    exhibits.append(
        ("Exhibit C \u2014 Waveform with flagged segments",
         waveform_png(artifacts.waveform, artifacts.sample_rate, result.result.segments, workdir / "wave.png"))
    )
    return exhibits


def generate_report(
    result: AnalysisResult,
    artifacts: AnalysisArtifacts | None,
    out_path: str | Path,
) -> Path:
    """Render ``result`` to a PDF at ``out_path`` and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    r = result.result
    meta = r.metadata
    baseline_only = result.provenance.baseline_only
    generated = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    story = []

    # --- Cover -------------------------------------------------------------
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("VoiceForensics", styles["VFTitle"]))
    story.append(Paragraph("Audio Authenticity \u2014 Forensic Analysis Report", styles["VFH1"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_kv_table(
        [
            ("Analysis ID", result.analysis_id),
            ("Generated", generated),
            ("Engine version", __version__),
            ("Analysis type", result.analysis_type.value),
        ],
        styles,
    ))
    if baseline_only:
        story.append(Spacer(1, 0.4 * cm))
        banner = Table([[Paragraph(f"<b>{templates.BASELINE_BANNER}</b>", styles["VFSmall"])]], colWidths=[16.5 * cm])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8c5")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4a72c")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(banner)

    # --- Findings ----------------------------------------------------------
    story.append(Paragraph("1. Findings", styles["VFH2"]))
    ci = r.confidence_interval
    verdict_color = _VERDICT_COLOR.get(r.verdict, colors.black)
    story.append(Paragraph(
        f'Verdict: <font color="{verdict_color.hexval()}"><b>{r.verdict.value}</b></font>',
        styles["VFBody"],
    ))
    story.append(_kv_table(
        [
            ("Deepfake probability", f"{r.deepfake_probability:.3f}"),
            ("Confidence interval", f"[{ci[0]:.3f}, {ci[1]:.3f}] (\u00b195%)"),
            ("Uncertainty (std)", f"{r.uncertainty:.3f}"),
            ("Naturalness score", "n/a" if r.naturalness_score is None else f"{r.naturalness_score:.3f}"),
        ],
        styles,
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(templates.verdict_interpretation(r.verdict), styles["VFBody"]))

    # --- Chain of custody --------------------------------------------------
    story.append(Paragraph("2. Chain of custody", styles["VFH2"]))
    ei = meta.edit_indicators
    edit_txt = "No edit indicators detected." if not ei.edited else "Possible processing detected: " + "; ".join(ei.reasons)
    story.append(_kv_table(
        [
            ("SHA-256", meta.file_hash_sha256),
            ("File size", f"{meta.size_bytes:,} bytes"),
            ("Duration", "n/a" if meta.duration_seconds is None else f"{meta.duration_seconds:.2f} s"),
            ("Container / codec", f"{meta.format or 'n/a'} / {meta.codec or 'n/a'}"),
            ("Sample rate / channels", f"{meta.sample_rate or 'n/a'} Hz / {meta.channels or 'n/a'}"),
            ("Edit indicators", edit_txt),
        ],
        styles,
    ))

    # --- Methodology -------------------------------------------------------
    story.append(Paragraph("3. Methodology", styles["VFH2"]))
    story.append(Paragraph(templates.METHODOLOGY, styles["VFBody"]))
    story.append(Spacer(1, 0.2 * cm))
    det_rows = [["Detector", "Available", "Prob.", "Weight"]]
    for s in result.provenance.detector_scores:
        det_rows.append([
            s.name, "yes" if s.available else "no",
            f"{s.prob_fake:.3f}" if s.available else "\u2014",
            f"{s.weight:.2f}",
        ])
    det_table = Table(det_rows, colWidths=[6 * cm, 3 * cm, 3.5 * cm, 4 * cm])
    det_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d1117")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(det_table)
    story.append(Paragraph(f"Active detectors: {', '.join(result.provenance.active_detectors)}.", styles["VFSmall"]))

    # --- Fingerprint -------------------------------------------------------
    if r.fingerprint is not None:
        story.append(Paragraph("4. Probable generation source (fingerprint)", styles["VFH2"]))
        fp = r.fingerprint
        story.append(_kv_table(
            [
                ("Probable source", fp.probable_source),
                ("Confidence", f"{fp.confidence:.3f}"),
                ("Alternatives", ", ".join(fp.alternative_sources) or "\u2014"),
            ],
            styles,
        ))
        story.append(Paragraph(
            "Source attribution uses a signature database of known generators. The shipped "
            "database contains placeholder priors and should be refined on a labelled corpus "
            "before being relied upon.", styles["VFSmall"],
        ))

    # --- Exhibits ----------------------------------------------------------
    workdir = Path(tempfile.mkdtemp(prefix="vf_report_"))
    exhibits = _render_exhibits(artifacts, result, workdir)
    if exhibits:
        story.append(PageBreak())
        story.append(Paragraph("5. Exhibits (annexures)", styles["VFH2"]))
        for caption, img_path in exhibits:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(caption, styles["VFBody"]))
            story.append(Image(str(img_path), width=16.5 * cm, height=16.5 * cm * 0.36))

    # --- Expert statement --------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("6. Expert witness statement", styles["VFH2"]))
    story.append(Paragraph(templates.EXPERT_STATEMENT, styles["VFBody"]))
    story.append(Spacer(1, 1.0 * cm))
    sign = Table(
        [["Signature:", "", "Date:", ""], ["Name:", "", "Designation:", ""]],
        colWidths=[2.5 * cm, 6 * cm, 2.5 * cm, 5.5 * cm],
    )
    sign.setStyle(TableStyle([
        ("LINEBELOW", (1, 0), (1, 0), 0.5, colors.black),
        ("LINEBELOW", (3, 0), (3, 0), 0.5, colors.black),
        ("LINEBELOW", (1, 1), (1, 1), 0.5, colors.black),
        ("LINEBELOW", (3, 1), (3, 1), 0.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(sign)

    # --- Appendix ----------------------------------------------------------
    story.append(Paragraph("7. Limitations", styles["VFH2"]))
    story.append(Paragraph(templates.LIMITATIONS, styles["VFBody"]))
    story.append(Paragraph("8. Glossary", styles["VFH2"]))
    for term, definition in templates.GLOSSARY.items():
        story.append(Paragraph(f"<b>{term}.</b> {definition}", styles["VFSmall"]))

    # --- Build -------------------------------------------------------------
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"VoiceForensics Report {result.analysis_id}", author="VoiceForensics",
        # Record the content hash + verdict in the (uncompressed) PDF metadata so the
        # report is self-describing and tamper-evidence travels with the file.
        subject=f"SHA-256:{meta.file_hash_sha256}",
        keywords=f"{result.analysis_id} {r.verdict.value} SHA-256:{meta.file_hash_sha256}",
    )
    footer = _footer_factory(meta.file_hash_sha256, baseline_only)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path
