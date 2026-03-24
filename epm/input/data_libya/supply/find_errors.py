import pandas as pd

file_path = "pAvailabilityCustom.csv"

# Lire le fichier en mode binaire pour repérer les caractères non ASCII
with open(file_path, "rb") as f:
    content = f.read()

# Identifier les caractères non ASCII
non_ascii_chars = set([chr(byte) for byte in content if byte >= 128])

print("Caractères problématiques détectés :", non_ascii_chars)
