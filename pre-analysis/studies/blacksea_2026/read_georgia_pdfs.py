import pdfplumber
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

files = {
    'Market2026': r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Georgian Power Market report March 2026.pdf',
    'GNERC2024':  r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\2024 GNERC Annual Report.pdf',
    'Gen2023':    r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\Generation Balance\generation_2023.pdf',
}

OUT = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\pdf_extract.txt'

with open(OUT, 'w', encoding='utf-8', errors='replace') as out:
    for name, path in files.items():
        out.write(f'\n{"="*60}\n')
        out.write(f'FILE: {name}\n')
        out.write(f'{"="*60}\n')
        try:
            with pdfplumber.open(path) as pdf:
                out.write(f'Total pages: {len(pdf.pages)}\n')
                for i, pg in enumerate(pdf.pages):
                    t = (pg.extract_text() or '').strip()
                    if not t:
                        continue
                    # Look for pages mentioning import/export/transit/cross-border
                    keywords = ['import', 'export', 'transit', 'azerbaijan', 'russia', 'turkey',
                                'armenia', 'cross-border', 'interconn', 'exchange', 'balance',
                                'GWh', 'TWh', 'transmission']
                    t_lower = t.lower()
                    relevant = any(k in t_lower for k in keywords)
                    if relevant or i < 5:
                        out.write(f'\n--- Page {i+1} ---\n')
                        out.write(t[:1500] + '\n')
        except Exception as e:
            out.write(f'ERROR: {e}\n')

print(f'Done. Output written to {OUT}')
