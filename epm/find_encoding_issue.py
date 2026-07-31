import os

for root, dirs, files in os.walk(r"C:\WBG\LocalITUtilities\EPM\epm\input\data_wapp"):
    for f in files:
        if f.endswith(".csv"):
            path = os.path.join(root, f)
            try:
                with open(path, encoding="utf-8") as fh:
                    fh.read()
            except UnicodeDecodeError as e:
                print("PROBLEME:", path, "->", e)