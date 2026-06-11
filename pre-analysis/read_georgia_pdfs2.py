import pdfplumber
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\pdf_extract2.txt'

# GNERC 2024 — scan all pages for import/export tables
GNERC = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\2024 GNERC Annual Report.pdf'
GEN23 = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\Generation Balance\generation_2023.pdf'

KEYWORDS = ['import', 'export', 'transit', 'azerbaijan', 'russia', 'turkey', 'armenia',
            'cross-border', 'interconnect', 'trade', 'balance', 'gwh', 'twh', 'mln kwh',
            'seasonal', 'summer', 'winter', 'quarter', 'monthly']

with open(OUT, 'w', encoding='utf-8', errors='replace') as f:
    for label, path in [('GNERC2024', GNERC), ('Gen2023', GEN23)]:
        f.write(f'\n{"="*60}\nFILE: {label}\n{"="*60}\n')
        try:
            with pdfplumber.open(path) as pdf:
                f.write(f'Total pages: {len(pdf.pages)}\n')
                for i, pg in enumerate(pdf.pages):
                    t = (pg.extract_text() or '').strip()
                    if not t:
                        continue
                    tl = t.lower()
                    if any(k in tl for k in KEYWORDS):
                        f.write(f'\n--- Page {i+1} ---\n')
                        f.write(t[:3000] + '\n')
        except Exception as e:
            f.write(f'ERROR: {e}\n')

print(f'Done -> {OUT}')
