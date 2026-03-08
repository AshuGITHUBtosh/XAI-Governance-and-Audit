from pathlib import Path
import json
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_audit_bundle(reports_dir: Path) -> dict:
    reports_dir = Path(reports_dir)
    bundle = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'artifacts': {}
    }
    files = {
        'dataset_profile': reports_dir / 'dataset_profile.json',
        'model_metrics':   reports_dir / 'model_metrics.json',
        'explainability':  reports_dir / 'explainability_summary.json',
        'fairness':        reports_dir / 'fairness_summary.json',
        'trajectory':      reports_dir / 'trajectory_summary.json',
    }
    for key, p in files.items():
        bundle['artifacts'][key] = load_json(p)
    return bundle


def save_audit_json(bundle: dict, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(bundle, f, indent=2)


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------

def save_audit_pdf(bundle: dict, out_pdf: Path):
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter
    margin = 50

    def new_page():
        c.showPage()
        return height - margin

    def check_y(y, needed=20):
        if y < margin + needed:
            return new_page()
        return y

    def section_header(y, text):
        y = check_y(y, 60)
        c.setFont('Helvetica-Bold', 13)
        c.setFillColorRGB(0.1, 0.2, 0.5)
        c.drawString(margin, y, text)
        c.setFillColorRGB(0, 0, 0)
        y -= 4
        c.setLineWidth(0.5)
        c.setStrokeColorRGB(0.1, 0.2, 0.5)
        c.line(margin, y, width - margin, y)
        c.setStrokeColorRGB(0, 0, 0)
        return y - 14

    def row(y, label, value, flag=None):
        y = check_y(y)
        c.setFont('Helvetica-Bold', 10)
        c.drawString(margin + 10, y, f"{label}:")
        c.setFont('Helvetica', 10)
        c.drawString(margin + 160, y, str(value))
        if flag:
            c.setFillColorRGB(0.8, 0.1, 0.1)
            c.drawString(margin + 300, y, flag)
            c.setFillColorRGB(0, 0, 0)
        return y - 14

    y = height - margin

    # ── Title ──────────────────────────────────────────────────────────────
    c.setFont('Helvetica-Bold', 18)
    c.setFillColorRGB(0.1, 0.2, 0.5)
    c.drawString(margin, y, 'ML Governance — Audit Report')
    c.setFillColorRGB(0, 0, 0)
    y -= 20
    c.setFont('Helvetica', 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(margin, y, f"Generated: {bundle.get('generated_at', 'N/A')}")
    c.setFillColorRGB(0, 0, 0)
    y -= 25

    # ── Dataset Profile ────────────────────────────────────────────────────
    dp = bundle['artifacts'].get('dataset_profile') or {}
    y = section_header(y, 'Dataset Profile')
    if dp:
        y = row(y, 'Rows',    dp.get('rows', 'N/A'))
        y = row(y, 'Columns', dp.get('columns', 'N/A'))
        y = row(y, 'Target',  dp.get('target', 'N/A'))

        # Target distribution
        target_counts = dp.get('target_counts', {})
        if target_counts:
            total = sum(target_counts.values())
            dist_str = '  |  '.join(
                f"{k}: {v} ({100*v/total:.1f}%)" for k, v in target_counts.items()
            )
            y = row(y, 'Class Distribution', dist_str)

        # Missingness
        miss = dp.get('missingness_percent_top', {})
        if miss:
            y = row(y, 'Missing Values', f"{len(miss)} columns affected")
            for k, v in list(miss.items())[:5]:
                y = check_y(y)
                c.setFont('Helvetica', 9)
                c.drawString(margin + 20, y, f"{k}: {v}%")
                y -= 12
        else:
            y = row(y, 'Missing Values', 'None detected ✓')

        # Suggested sensitive columns
        sens_cols = dp.get('suggested_sensitive_columns', [])
        if sens_cols:
            y = row(y, 'Sensitive Columns', ', '.join(sens_cols))
    else:
        c.setFont('Helvetica', 10)
        c.drawString(margin + 10, y, 'No dataset profile found.')
        y -= 14

    y -= 10

    # ── Model Metrics ──────────────────────────────────────────────────────
    mm = bundle['artifacts'].get('model_metrics') or {}
    y = section_header(y, 'Model Performance Metrics')
    if mm:
        if mm.get('accuracy') is not None:
            y = row(y, 'Accuracy',          f"{float(mm['accuracy']):.4f}")
        if mm.get('tuned_accuracy') is not None:
            y = row(y, 'Tuned Accuracy',    f"{float(mm['tuned_accuracy']):.4f}")
        if mm.get('roc_auc') is not None:
            auc = float(mm['roc_auc'])
            flag = '⚠ Below 0.75' if auc < 0.75 else '✓ Acceptable'
            y = row(y, 'ROC AUC',           f"{auc:.4f}", flag)
        if mm.get('optimal_threshold') is not None:
            y = row(y, 'Optimal Threshold', f"{float(mm['optimal_threshold']):.4f}")

        # Classification report summary
        cr = mm.get('classification_report', {})
        if cr:
            y -= 4
            y = check_y(y, 40)
            c.setFont('Helvetica-Bold', 10)
            c.drawString(margin + 10, y, 'Classification Report (weighted avg):')
            y -= 14
            wa = cr.get('weighted avg', {})
            if wa:
                c.setFont('Helvetica', 10)
                summary = (f"Precision: {wa.get('precision',0):.3f}  |  "
                           f"Recall: {wa.get('recall',0):.3f}  |  "
                           f"F1: {wa.get('f1-score',0):.3f}")
                c.drawString(margin + 20, y, summary)
                y -= 14
    else:
        c.setFont('Helvetica', 10)
        c.drawString(margin + 10, y, 'No model metrics found.')
        y -= 14

    y -= 10

    # ── Fairness ───────────────────────────────────────────────────────────
    fa = bundle['artifacts'].get('fairness') or {}
    y = section_header(y, 'Fairness Assessment')
    if fa:
        dp_val = fa.get('demographic_parity_difference', None)
        eo_val = fa.get('equalized_odds_difference', None)

        if dp_val is not None:
            dp_f = float(dp_val)
            flag = '⚠ HIGH — exceeds 0.20 limit' if dp_f > 0.2 else '✓ Within limit'
            y = row(y, 'Demographic Parity Diff', f"{dp_f:.4f}", flag)

        if eo_val is not None:
            eo_f = float(eo_val)
            flag = '⚠ HIGH — exceeds 0.20 limit' if eo_f > 0.2 else '✓ Within limit'
            y = row(y, 'Equalized Odds Diff',     f"{eo_f:.4f}", flag)

        # Interpretation note
        y -= 4
        y = check_y(y, 30)
        c.setFont('Helvetica-Oblique', 9)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        c.drawString(margin + 10, y,
            'Note: Values > 0.20 indicate potential bias requiring mitigation review.')
        c.setFillColorRGB(0, 0, 0)
        y -= 14
    else:
        c.setFont('Helvetica', 10)
        c.drawString(margin + 10, y, 'No fairness summary found.')
        y -= 14

    y -= 10

    # ── Explainability ─────────────────────────────────────────────────────
    ex = bundle['artifacts'].get('explainability') or {}
    y = section_header(y, 'Explainability — Top SHAP Features')
    if ex and ex.get('mean_abs_shap'):
        items = sorted(ex['mean_abs_shap'].items(), key=lambda x: x[1], reverse=True)[:10]
        max_val = items[0][1] if items else 1

        for rank, (feat, val) in enumerate(items, 1):
            y = check_y(y, 16)
            c.setFont('Helvetica', 9)

            # Draw bar proportional to importance
            bar_max = 120
            bar_len = int((val / max_val) * bar_max)
            c.setFillColorRGB(0.2, 0.4, 0.7)
            c.rect(margin + 180, y - 2, bar_len, 10, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)

            c.setFont('Helvetica', 9)
            c.drawString(margin + 10, y, f"{rank:2}. {feat}")
            c.drawString(margin + 310, y, f"{val:.4f}")
            y -= 14
    else:
        c.setFont('Helvetica', 10)
        c.drawString(margin + 10, y, 'No explainability summary found.')
        y -= 14

    y -= 10

    # ── Trajectory ─────────────────────────────────────────────────────────
    tr = bundle['artifacts'].get('trajectory') or {}
    y = section_header(y, 'Trajectory & Drift Summary')
    if tr:
        # PSI
        psi = tr.get('psi', {}) or {}
        if psi:
            valid_psi = [(k, v) for k, v in psi.items() if v is not None]
            high_drift = [(k, v) for k, v in valid_psi if float(v) > 0.2]
            mod_drift  = [(k, v) for k, v in valid_psi if 0.1 < float(v) <= 0.2]

            y = row(y, 'Features Monitored', len(valid_psi))
            y = row(y, 'High Drift (PSI>0.2)',
                    f"{len(high_drift)} features" if high_drift else 'None ✓')
            y = row(y, 'Moderate Drift (PSI>0.1)',
                    f"{len(mod_drift)} features" if mod_drift else 'None ✓')

            if high_drift or mod_drift:
                y -= 4
                c.setFont('Helvetica-Bold', 9)
                c.drawString(margin + 10, y, 'Top drifted features:')
                y -= 12
                for k, v in sorted(valid_psi, key=lambda x: x[1], reverse=True)[:5]:
                    y = check_y(y)
                    flag = ' ⚠ HIGH' if float(v) > 0.2 else (' ⚠ MOD' if float(v) > 0.1 else '')
                    c.setFont('Helvetica', 9)
                    c.drawString(margin + 20, y, f"{k}: {float(v):.4f}{flag}")
                    y -= 12

        # ✅ Fixed: bucket performance with age bins not raw ages
        bp = tr.get('bucket_performance') or []
        if bp:
            y -= 6
            y = check_y(y, 40)
            c.setFont('Helvetica-Bold', 10)
            c.drawString(margin + 10, y, 'Performance by sensitive group:')
            y -= 14

            # Filter out groups with n < 5 (statistically unreliable)
            meaningful = [r for r in bp if (r.get('count') or 0) >= 5]
            if not meaningful:
                meaningful = bp  # show all if none meet threshold

            for rec in meaningful:
                y = check_y(y)
                group = rec.get('sensitive',
                        rec.get('age_bucket',
                        str(list(rec.values())[0])))
                count = rec.get('count', 'N/A')
                acc   = rec.get('accuracy', 'N/A')
                try:
                    acc_str = f"{float(acc):.3f}"
                    acc_flag = ' ⚠' if float(acc) < 0.65 else ''
                except (TypeError, ValueError):
                    acc_str = str(acc)
                    acc_flag = ''
                c.setFont('Helvetica', 9)
                c.drawString(margin + 20, y,
                    f"Group '{group}': n={count}, accuracy={acc_str}{acc_flag}")
                y -= 12
    else:
        c.setFont('Helvetica', 10)
        c.drawString(margin + 10, y, 'No trajectory summary found.')
        y -= 14

    # ── Footer ─────────────────────────────────────────────────────────────
    c.showPage()
    c.setFont('Helvetica', 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(margin, 30,
        'This report was generated automatically by the ML Governance Toolkit. '
        'For regulatory review only.')
    c.setFillColorRGB(0, 0, 0)
    c.save()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    reports_dir = Path('reports')
    out_json = reports_dir / 'audit_report.json'
    out_pdf  = reports_dir / 'audit_report.pdf'
    bundle = build_audit_bundle(reports_dir)
    save_audit_json(bundle, out_json)
    save_audit_pdf(bundle, out_pdf)
    print(f'Wrote audit JSON to {out_json} and PDF to {out_pdf}')


if __name__ == '__main__':
    main()