"""
Extract monthly cross-border import flows from generation_YYYY.pdf files.
Each PDF has 3 pages: Page1=Jan-Apr, Page2=May-Aug, Page3=Sep-Dec (sometimes Sep-Annual).
"""
import pdfplumber, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = r'C:\Users\wb590892\Documents\EPM_Models\black_sea_2026\Data\Georgia\Team\Generation Balance'

# Month order per page
PAGE_MONTHS = {
    1: ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL'],
    2: ['MAY', 'JUNE', 'JULY', 'AUGUST'],
    3: ['SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER'],
}

ROWS_OF_INTEREST = [
    'Total Import',
    'Import from Azerbaijan',
    'Import from Russia',
    'Import from Turkey',
    'Import from Armenia',
    'Total Transit',
    'Transit from Azerbaijan',
    'Transit from Russia',
]

def classify_row(label):
    ll = label.lower()
    for r in ROWS_OF_INTEREST:
        if r.lower() in ll:
            return r
    return None

results = {}  # year -> month -> {row: value}

for yr in ['2020', '2021', '2022', '2023']:
    path = os.path.join(BASE, f'generation_{yr}.pdf')
    if not os.path.exists(path):
        continue
    results[yr] = {}

    with pdfplumber.open(path) as pdf:
        for page_i, pg in enumerate(pdf.pages, start=1):
            months = PAGE_MONTHS.get(page_i, [])

            # Try table extraction first
            tables = pg.extract_tables()
            if not tables:
                continue

            # Use the first table (main balance)
            tbl = tables[0]
            if not tbl:
                continue

            # Find month columns from header row
            # Row 0 should be header with month names
            # Columns: #, TITLE, [MONTH_TOTALLY_SOLD, D/C, BALANCING] x 4
            # We want TOTALLY_SOLD = col 2, 5, 8, 11 (0-indexed after TITLE)
            month_cols = []  # list of (month_name, col_index)

            # Detect month columns from row 0
            header_row = tbl[0] if tbl else []
            for ci, cell in enumerate(header_row):
                if not cell:
                    continue
                cell_up = str(cell).upper().strip()
                # Match any month name
                for m in ['JANUARY', 'FEBERUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
                          'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']:
                    if m in cell_up:
                        norm = m.replace('FEBERUARY', 'FEBRUARY')
                        month_cols.append((norm, ci))
                        break

            # If no month headers found in row 0, check row 1
            if not month_cols and len(tbl) > 1:
                for ci, cell in enumerate(tbl[1]):
                    if not cell: continue
                    cell_up = str(cell).upper().strip()
                    for m in ['JANUARY','FEBERUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE',
                              'JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']:
                        if m in cell_up:
                            norm = m.replace('FEBERUARY', 'FEBRUARY')
                            month_cols.append((norm, ci))
                            break

            # For each month, TOTALLY SOLD is the first sub-col (same col as month header, or +0)
            # Typically: col=2 for first month, col=5 for second, etc. (step of 3)
            # But we derive from month_col positions: TOTALLY_SOLD = month_col itself

            # Now scan rows for our rows of interest
            for row in tbl[2:]:  # skip 2 header rows
                if not row or not row[1]:
                    continue
                label = str(row[1]).strip()
                key = classify_row(label)
                if not key:
                    continue

                for month_name, mc in month_cols:
                    # TOTALLY SOLD = value at mc (first sub-col)
                    val = row[mc] if mc < len(row) else None
                    try:
                        v = float(str(val).replace(' ', '').replace(',', '.')) if val else 0.0
                    except:
                        v = 0.0

                    if month_name not in results[yr]:
                        results[yr][month_name] = {}

                    existing = results[yr][month_name].get(key, 0.0)
                    results[yr][month_name][key] = existing + v  # sum across traders

# Print results
MONTH_ORDER = ['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE',
               'JULY','AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER']

print(f"\n{'Year':<6} {'Month':<12} {'TotImport':>10} {'AZ':>8} {'RU':>8} {'TR':>8} {'ARM':>8} {'Transit':>10}")
print('-'*75)

for yr in sorted(results.keys()):
    for month in MONTH_ORDER:
        d = results[yr].get(month, {})
        if not d:
            continue
        ti = d.get('Total Import', 0)
        az = d.get('Import from Azerbaijan', 0)
        ru = d.get('Import from Russia', 0)
        tr = d.get('Import from Turkey', 0)
        arm = d.get('Import from Armenia', 0)
        tr_az = d.get('Transit from Azerbaijan', 0) + d.get('Total Transit', 0)

        print(f"{yr:<6} {month[:9]:<12} {ti:>10.1f} {az:>8.1f} {ru:>8.1f} {tr:>8.1f} {arm:>8.1f} {tr_az:>10.1f}")
    print()

# Also print seasonal totals
print("\n=== SEASONAL TOTALS (GWh) ===")
SUMMER = ['APRIL','MAY','JUNE','JULY','AUGUST','SEPTEMBER']
WINTER = ['JANUARY','FEBRUARY','MARCH','OCTOBER','NOVEMBER','DECEMBER']

print(f"{'Year':<6} {'Season':<8} {'TotImport':>10} {'AZ':>8} {'RU':>8} {'TR':>8} {'ARM':>8}")
print('-'*60)
for yr in sorted(results.keys()):
    for season, months in [('SUMMER', SUMMER), ('WINTER', WINTER)]:
        ti = az = ru = tr = arm = 0
        for m in months:
            d = results[yr].get(m, {})
            ti  += d.get('Total Import', 0)
            az  += d.get('Import from Azerbaijan', 0)
            ru  += d.get('Import from Russia', 0)
            tr  += d.get('Import from Turkey', 0)
            arm += d.get('Import from Armenia', 0)
        print(f"{yr:<6} {season:<8} {ti:>10.1f} {az:>8.1f} {ru:>8.1f} {tr:>8.1f} {arm:>8.1f}")
