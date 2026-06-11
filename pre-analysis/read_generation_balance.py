import pdfplumber
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Read all generation_YYYY.pdf files and extract key cross-border rows
import os

OUT = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\gen_balance_extract.txt'
BASE = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\Generation Balance'

KEYWORDS = ['import', 'export', 'transit', 'azerbaijan', 'russia', 'turkey', 'armenia']

with open(OUT, 'w', encoding='utf-8', errors='replace') as f:
    for yr in ['2020', '2021', '2022', '2023']:
        path = os.path.join(BASE, f'generation_{yr}.pdf')
        if not os.path.exists(path):
            continue
        f.write(f'\n{"="*60}\nYEAR: {yr}\n{"="*60}\n')
        with pdfplumber.open(path) as pdf:
            f.write(f'Pages: {len(pdf.pages)}\n')
            for i, pg in enumerate(pdf.pages):
                t = (pg.extract_text() or '').strip()
                tl = t.lower()
                if any(k in tl for k in KEYWORDS):
                    f.write(f'\n--- Page {i+1} text ---\n')
                    # Print relevant lines only
                    for line in t.split('\n'):
                        ll = line.lower()
                        if any(k in ll for k in KEYWORDS) or 'total import' in ll or 'total export' in ll or 'transit' in ll:
                            f.write(line + '\n')

                # Try table extraction
                tables = pg.extract_tables()
                if tables:
                    for ti, tbl in enumerate(tables):
                        has_xborder = False
                        for row in tbl:
                            if row and any(k in str(row).lower() for k in KEYWORDS):
                                has_xborder = True
                                break
                        if has_xborder:
                            f.write(f'\n--- Page {i+1} table {ti} ---\n')
                            for row in tbl:
                                if row:
                                    row_str = [str(c or '').strip()[:30] for c in row]
                                    f.write(' | '.join(row_str) + '\n')

print(f'Done -> {OUT}')
