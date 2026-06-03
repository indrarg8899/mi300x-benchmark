"""Report generation."""
import json

def generate_html(results, output_path="report.html"):
    rows = ""
    for r in results:
        d = r.__dict__ if hasattr(r, '__dict__') else r
        rows += f"<tr><td>{d['name']}</td><td>{d['value']:.2f}</td><td>{d['unit']}</td></tr>\n"
    html = f"""<!DOCTYPE html><html><head><title>MI300X Benchmark Report</title>
    <style>body{{font-family:sans-serif;margin:40px}}table{{border-collapse:collapse;width:100%}}
    th,td{{border:1px solid #ddd;padding:8px}}th{{background:#f4f4f4}}</style></head>
    <body><h1>MI300X Benchmark Report</h1><table><tr><th>Benchmark</th><th>Value</th><th>Unit</th></tr>
    {rows}</table></body></html>"""
    with open(output_path, "w") as f:
        f.write(html)

def generate_markdown(results, output_path="REPORT.md"):
    lines = ["# MI300X Benchmark Report\n", "| Benchmark | Value | Unit |", "|-----------|-------|------|"]
    for r in results:
        d = r.__dict__ if hasattr(r, '__dict__') else r
        lines.append(f"| {d['name']} | {d['value']:.2f} | {d['unit']} |")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
