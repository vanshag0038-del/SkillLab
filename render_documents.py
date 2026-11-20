"""Render the final DOCX content to PDF through a local HTML intermediate.

Uses the installed Edge engine with an isolated headless profile. No document
content leaves the computer. This fallback avoids desktop Office/LibreOffice.
"""
import base64
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
QA = ROOT / 'outputs/final-qa'


def paragraph_html(para, doc):
    if para._p.xpath('.//w:br[@w:type="page"]'):
        return '<div class="pagebreak"></div>'
    images = []
    for drawing in para._p.xpath('.//w:drawing'):
        blip = drawing.xpath('.//a:blip')[0]
        part = doc.part.related_parts[blip.get(qn('r:embed'))]
        extent = drawing.xpath('.//wp:extent')[0]
        width = int(extent.get('cx')) / 914400
        data = base64.b64encode(part.blob).decode('ascii')
        images.append(f'<img style="width:{width:.3f}in" src="data:{part.content_type};base64,{data}">')
    if images:
        return '<p class="figure">' + ''.join(images) + '</p>'
    value = html.escape(para.text)
    style = para.style.name
    if style == 'Title':
        return f'<h1 class="title">{value}</h1>'
    if style.startswith('Heading'):
        level = int(style.split()[-1]) + 1
        return f'<h{level}>{value}</h{level}>'
    if style == 'Subtitle':
        return f'<p class="subtitle">{value}</p>'
    monospace = para.runs and para.runs[0].font.name == 'Consolas'
    return f'<p class="code">{value or "&nbsp;"}</p>' if monospace else f'<p>{value or "&nbsp;"}</p>'


def as_html(path):
    doc = Document(path)
    blocks = []
    expected_text = []
    for child in doc.element.body:
        if child.tag == qn('w:p'):
            para = Paragraph(child, doc)
            blocks.append(paragraph_html(para, doc))
            expected_text.append(para.text)
        elif child.tag == qn('w:tbl'):
            table = Table(child, doc)
            rows = []
            for number, row in enumerate(table.rows):
                cells = []
                for cell in row.cells:
                    tag = 'th' if number == 0 else 'td'
                    width = (cell.width / 914400) if cell.width else 1
                    value = '<br>'.join(html.escape(p.text) for p in cell.paragraphs)
                    expected_text.extend(p.text for p in cell.paragraphs)
                    cells.append(f'<{tag} style="width:{width:.3f}in">{value}</{tag}>')
                rows.append('<tr>' + ''.join(cells) + '</tr>')
            blocks.append('<table>' + ''.join(rows) + '</table>')
    css = '''
    @page {size:A4; margin:0.62in 0.7in 0.62in 0.7in;}
    body{font:10.5pt Calibri,Arial,sans-serif;line-height:1.05;color:#000;margin:0;}
    p{margin:0 0 6pt;orphans:2;widows:2;}
    h1,h2,h3{color:#000;break-after:avoid;page-break-after:avoid;font-weight:bold;}
    h1.title{font-size:25pt;line-height:1.12;font-weight:400;margin:0 0 10pt;}
    h2{font-size:17pt;line-height:1.12;margin:0 0 9pt;}
    h3{font-size:12pt;line-height:1.1;margin:10pt 0 6pt;}
    p.subtitle{font-size:13pt;margin:0 0 12pt;}
    .code{font:7.7pt Consolas,monospace;line-height:1;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;}
    .pagebreak{break-before:page;page-break-before:always;height:0;}
    .figure{break-inside:avoid;line-height:0;margin:0 0 8pt;}
    img{max-width:100%;height:auto;}
    table{border-collapse:collapse;table-layout:fixed;width:6.7in;margin:8pt 0 10pt;font-size:9pt;line-height:1.12;}
    td,th{border:0.6pt solid #d9d9d9;padding:5pt 6pt;vertical-align:middle;overflow-wrap:anywhere;text-align:left;}
    th{background:#e7eef4;}tr{break-inside:avoid;page-break-inside:avoid;}
    '''
    return '<!doctype html><html><head><meta charset="utf-8"><style>' + css + '</style></head><body>' + ''.join(blocks) + '</body></html>', expected_text


def main():
    QA.mkdir(parents=True, exist_ok=True)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'renderer': 'DOCX content and layout mapped to local HTML; headless Microsoft Edge PDF; PyMuPDF page rasterization',
              'documents': {}}
    with sync_playwright() as play:
        browser = play.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page()
        for stem in ['Completed_Lab_Record', 'Completed_Quiz_Answers']:
            source = ROOT / f'{stem}.docx'
            markup, expected = as_html(source)
            (QA / f'{stem}.html').write_text(markup, encoding='utf-8')
            page.set_content(markup, wait_until='load')
            page.evaluate('document.fonts.ready')
            page.pdf(path=str(ROOT / f'{stem}.pdf'), print_background=True,
                     prefer_css_page_size=True, display_header_footer=True,
                     header_template='<span></span>',
                     footer_template='<div style="font:8px Arial;width:100%;text-align:right;padding-right:67px">1MS25CI051 | <span class="pageNumber"></span></div>')
            rendered = pymupdf.open(ROOT / f'{stem}.pdf')
            all_text = ' '.join(p.get_text() for p in rendered)
            normalized = ''.join(all_text.split())
            for value in expected:
                assert ''.join(value.split()) in normalized, f'Text missing in PDF: {value[:100]}'
            folder = QA / stem
            folder.mkdir(exist_ok=True)
            for number, pdf_page in enumerate(rendered, 1):
                assert pdf_page.get_text().strip(), f'Blank page {number}'
                pdf_page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(folder / f'page-{number:02}.png')
                for x0,y0,x1,y1,*_ in pdf_page.get_text('blocks'):
                    assert x0 >= 0 and y0 >= 0 and x1 <= pdf_page.rect.width + 1 and y1 <= pdf_page.rect.height + 1, f'Out-of-page text at {number}'
            for extension in ['docx', 'pdf']:
                path = ROOT / f'{stem}.{extension}'
                report['documents'][path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'pages': len(rendered)}
            print(f'{stem}: {len(rendered)} pages; all source text present; page images created')
        browser.close()
    (ROOT / 'outputs/document_qa.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
