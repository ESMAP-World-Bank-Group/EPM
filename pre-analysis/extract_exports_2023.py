"""
Check if generation_2023.pdf has an export section (page 3/4).
Also check generation_2022.pdf for December data.
"""
import pdfplumber, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\Generation Balance'

for yr in ['2022', '2023']:
    path = os.path.join(BASE, f'generation_{yr}.pdf')
    print(f'\n{"="*60}\nYEAR: {yr}\n{"="*60}')
    with pdfplumber.open(path) as pdf:
        print(f'Pages: {len(pdf.pages)}')
        for i, pg in enumerate(pdf.pages, 1):
            t = (pg.extract_text() or '').strip()
            # Show page header
            first_line = t.split('\n')[0] if t else '(empty)'
            print(f'\nPage {i}: {first_line[:80]}')
            # Look for export/total rows
            for line in t.split('\n'):
                ll = line.lower()
                if any(k in ll for k in ['export', 'total sold', 'total import', 'total transit',
                                          'transit', 'september', 'october', 'november', 'december']):
                    print(f'  > {line[:100]}')
