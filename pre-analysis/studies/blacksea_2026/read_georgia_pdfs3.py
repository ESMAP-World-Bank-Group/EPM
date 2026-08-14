import pdfplumber
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\pdf_extract3.txt'

GNERC = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\2024 GNERC Annual Report.pdf'
GEN23 = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\Generation Balance\generation_2023.pdf'

with open(OUT, 'w', encoding='utf-8', errors='replace') as f:
    # GNERC pages 9-14 (infographic + import/export detail)
    f.write('=== GNERC 2024 pages 9-14 ===\n')
    with pdfplumber.open(GNERC) as pdf:
        f.write(f'Total pages: {len(pdf.pages)}\n')
        for i in [8, 9, 10, 11, 12, 13]:  # 0-indexed = pages 9-14
            pg = pdf.pages[i]
            t = (pg.extract_text() or '').strip()
            f.write(f'\n--- Page {i+1} ---\n{t}\n')
            # Also try to extract tables
            tables = pg.extract_tables()
            if tables:
                f.write(f'  TABLES ({len(tables)}):\n')
                for ti, tbl in enumerate(tables):
                    f.write(f'  Table {ti}:\n')
                    for row in tbl:
                        f.write('    ' + ' | '.join(str(c or '') for c in row) + '\n')

    # generation_2023.pdf — all pages
    f.write('\n\n=== generation_2023.pdf ===\n')
    with pdfplumber.open(GEN23) as pdf:
        f.write(f'Total pages: {len(pdf.pages)}\n')
        for i, pg in enumerate(pdf.pages):
            t = (pg.extract_text() or '').strip()
            if t:
                f.write(f'\n--- Page {i+1} ---\n{t[:2000]}\n')
            tables = pg.extract_tables()
            if tables:
                f.write(f'  TABLES ({len(tables)}):\n')
                for ti, tbl in enumerate(tables):
                    f.write(f'  Table {ti}:\n')
                    for row in tbl[:20]:
                        f.write('    ' + ' | '.join(str(c or '') for c in row) + '\n')

print(f'Done -> {OUT}')
