import pandas as pd


def find_ascii(file):

    file_path = file
    with open(file_path, "rb") as f:
        content = f.read()
        
    non_ascii_chars = set([chr(byte) for byte in content if byte >= 128])
    print("Problematic items :", non_ascii_chars)

    def find_non_ascii(s):
        return ''.join(c for c in str(s) if ord(c) >= 128)  # Garde uniquement les caractères problématiques
    
    df = pd.read_csv(file_path, encoding="ISO-8859-1", errors="replace")  # Remplace les erreurs par "?"
    df['problem_chars'] = df['gen'].apply(find_non_ascii)
    problematic_gens = df[df['problem_chars'] != ""][['gen', 'problem_chars']]
    print(problematic_gens)



def find_duplicates(file):

    file_path = file  
    df = pd.read_csv(file_path)

    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        print("Duplicates in:", file_path)
        print(duplicates)
    else:
        print("No duplicates in:", file_path)

