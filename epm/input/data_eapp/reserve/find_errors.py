import pandas as pd


def find_ascii(file):

    file_path = file
    with open(file_path, "rb") as f:
        content = f.read()

    non_ascii_chars = set([chr(byte) for byte in content if byte >= 128])
    print("Problematic items :", non_ascii_chars)

    def find_non_ascii(s):
        return ''.join(c for c in str(s) if ord(c) >= 128)  # Garde uniquement les caractères problématiques
    
    df = pd.read_csv(file_path, encoding="ISO-8859-1")  # Remplace les erreurs par "?"
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

def check_techs(var='zone'):
    # Charger les fichiers CSV

    df_GenDataExcel = pd.read_csv("../supply/pGenDataExcelCustom.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire
    df_GenDataExcel_NaN = df_GenDataExcel.loc[(df_GenDataExcel['gen'].isna())|(df_GenDataExcel['zone'].isna())|(df_GenDataExcel['fuel'].isna())|(df_GenDataExcel['tech'].isna())]

    if var == 'zone':
        df_tech_list = pd.read_csv("../zcmap.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire
        df_data = pd.read_csv("../supply/pGenDataExcelCustom.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire

        # Extraire la liste des technologies du premier fichier
        tech_list = set(df_tech_list["Zone"])

        # Vérifier si chaque technologie du deuxième fichier est présente dans la liste
        df_data["Tech Valid"] = df_data["zone"].isin(tech_list)

        # Afficher les technologies non reconnues
        invalid_techs = df_data.loc[~df_data["Tech Valid"], ["zone"]].drop_duplicates()

    elif var =='tech':
        df_tech_list = pd.read_csv("../resources/pTechData.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire
        df_data = pd.read_csv("../supply/pGenDataExcelCustom.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire

        # Extraire la liste des technologies du premier fichier
        tech_list = set(df_tech_list["Technology"])

        # Vérifier si chaque technologie du deuxième fichier est présente dans la liste
        df_data["Tech Valid"] = df_data["tech"].isin(tech_list)

        # Afficher les technologies non reconnues
        invalid_techs = df_data.loc[~df_data["Tech Valid"], ["tech"]].drop_duplicates()
    
    elif var =='fuel':
        df_tech_list = pd.read_csv("../resources/ftfindex.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire
        df_data = pd.read_csv("../supply/pGenDataExcelCustom.csv", delimiter=",")  # Ajuste le délimiteur si nécessaire

        # Extraire la liste des technologies du premier fichier
        tech_list = set(df_tech_list["Fuel"])

        # Vérifier si chaque technologie du deuxième fichier est présente dans la liste
        df_data["Tech Valid"] = df_data["fuel"].isin(tech_list)

        # Afficher les technologies non reconnues
        invalid_techs = df_data.loc[~df_data["Tech Valid"], ["fuel"]].drop_duplicates()

    # Affichage des résultats
    if invalid_techs.empty:
        print("Toutes les technologies du deuxième fichier sont présentes dans le premier fichier.")
    else:
        print("Technologies non reconnues:")
        print(invalid_techs)

    # Sauvegarde du fichier mis à jour si besoin

    return df_GenDataExcel_NaN