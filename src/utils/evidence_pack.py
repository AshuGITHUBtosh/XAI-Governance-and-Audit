from pathlib import Path
import json
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_audit_bundle(out_json: Path) -> dict:
    bundle = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'artifacts': {}
    }

    files = {
        'dataset_profile': Path('reports/dataset_profile.json'),
        'model_metrics': Path('reports/model_metrics.json'),
        'explainability': Path('reports/explainability_summary.json'),
        'fairness': Path('reports/fairness_summary.json'),
        'trajectory': Path('reports/trajectory_summary.json')
    }

    for key, p in files.items():
        data = load_json(p)
        bundle['artifacts'][key] = data

    return bundle


def save_audit_json(bundle: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2)


def save_audit_pdf(bundle: dict, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter

    margin = 50
    y = height - margin
    c.setFont('Helvetica-Bold', 16)
    c.drawString(margin, y, 'Audit Report')
    y -= 30
    c.setFont('Helvetica', 10)
    c.drawString(margin, y, f"Generated at: {bundle.get('generated_at')}")
    y -= 25

    dp = bundle['artifacts'].get('dataset_profile') or {}
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Dataset Profile:')
    y -= 18
    c.setFont('Helvetica', 10)
    if dp:
        rows = dp.get('rows')
        cols = dp.get('columns')
        c.drawString(margin, y, f"Rows: {rows}  Columns: {cols}")
        y -= 14
        miss = dp.get('missingness_percent_top', {})
        if miss:
            c.drawString(margin, y, 'Top missingness:')
            y -= 12
            for k, v in list(miss.items())[:5]:
                c.drawString(margin + 10, y, f"{k}: {v}%")
                y -= 12
    else:
        c.drawString(margin, y, 'No dataset profile found')
        y -= 14

    y -= 6
    mm = bundle['artifacts'].get('model_metrics') or {}
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Model Metrics:')
    y -= 18
    c.setFont('Helvetica', 10)
    if mm:
        c.drawString(margin, y, f"Accuracy: {mm.get('accuracy')}")
        y -= 12
        if 'roc_auc' in mm:
            c.drawString(margin, y, f"ROC AUC: {mm.get('roc_auc')}")
            y -= 12
    else:
        c.drawString(margin, y, 'No model metrics found')
        y -= 12

    y -= 6
    ex = bundle['artifacts'].get('explainability') or {}
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Explainability (Top features):')
    y -= 18
    c.setFont('Helvetica', 10)
    if ex and ex.get('mean_abs_shap'):
        items = sorted(ex['mean_abs_shap'].items(), key=lambda x: x[1], reverse=True)[:10]
        for k, v in items:
            c.drawString(margin, y, f"{k}: {v:.4f}")
            y -= 12
            if y < margin + 50:
                c.showPage()
                y = height - margin
    else:
        c.drawString(margin, y, 'No explainability summary found')
        y -= 12

    y -= 6
    fa = bundle['artifacts'].get('fairness') or {}
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Fairness Summary:')
    y -= 18
    c.setFont('Helvetica', 10)
    if fa:
        for k, v in fa.items():
            c.drawString(margin, y, f"{k}: {v}")
            y -= 12
            if y < margin + 50:
                c.showPage()
                y = height - margin
    else:
        c.drawString(margin, y, 'No fairness summary found')
        y -= 12

    # Trajectory summary (PSI, bucket perf)
    tr = bundle['artifacts'].get('trajectory') or {}
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, 'Trajectory Summary:')
    y -= 18
    c.setFont('Helvetica', 10)
    if tr:
        psi = tr.get('psi', {}) or {}
        if psi:
            c.drawString(margin, y, 'Top PSI features:')
            y -= 14
            items = sorted(psi.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)[:8]
            for k, v in items:
                try:
                    val = float(v) if v is not None else None
                except Exception:
                    val = v
                c.drawString(margin + 6, y, f"{k}: {val}")
                y -= 12
                if y < margin + 50:
                    c.showPage()
                    y = height - margin
        bp = tr.get('bucket_performance') or []
        if bp:
            c.drawString(margin, y, 'Bucket performance (count / accuracy):')
            y -= 14
            for rec in bp:
                line = f"{rec.get('age_bucket', rec.get('index', 'bucket'))}: {rec.get('count')} / {rec.get('accuracy')}"
                c.drawString(margin + 6, y, line)
                y -= 12
                if y < margin + 50:
                    c.showPage()
                    y = height - margin
    else:
        c.drawString(margin, y, 'No trajectory summary found')
        y -= 12

    c.showPage()
    c.save()


def main():
    out_json = Path('reports/audit_report.json')
    out_pdf = Path('reports/audit_report.pdf')
    bundle = build_audit_bundle(out_json)
    save_audit_json(bundle, out_json)
    save_audit_pdf(bundle, out_pdf)
    print(f'Wrote audit JSON to {out_json} and PDF to {out_pdf}')


if __name__ == '__main__':
    main()
