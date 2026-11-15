"""Pasos 3 y 4 — Selección de los días representativos.

OBJETO
    Reducir un año completo de datos horarios — 8 760 horas, 9 regiones, tres
    series — a un número pequeño de días tipo ponderados, y luego ponerlos en
    el formato de EPM.

ENTRADAS
    input/load.csv   demanda horaria por GCR, paso 1, fuente CENACE
    input/pv.csv     factor de carga solar, paso 2, fuente Renewables.ninja
    input/wind.csv   factor de carga eólico, ídem

ESTACIONES
    Cuatro estaciones alineadas con los trimestres civiles. El corte trimestral
    tiene sentido físico en México, donde la punta es claramente estival, y
    coincide con la estacionalidad que publica el PLADESE.
    No se usa entonces `representativeseasons_pipeline.py`, que deduciría las
    estaciones por agrupamiento estadístico: un corte legible y reproducible
    vale más aquí que un corte óptimo pero opaco.

NÚMERO DE DÍAS
    Tres días optimizados por estación, a los que el pipeline añade
    automáticamente los días extremos — insolación mínima, viento mínimo,
    demanda máxima. Es decir, del orden de veinticuatro días tipo.

    Este número es impuesto, no elegido por el modelo: la restricción del
    programa de optimización es una igualdad estricta sobre el número de días
    por estación. Tres es un compromiso — con un solo día optimizado, los días
    extremos cargarían las tres cuartas partes del peso anual y el sistema
    quedaría dimensionado sobre situaciones atípicas.

SALIDAS
    output/epm/pHours.csv          peso de cada día, en horas
    output/epm/pDemandProfile.csv  demanda normalizada
    output/epm/pVREProfile.csv     factores de carga solar y eólico
    output/representative_days_summary.csv   días retenidos y su naturaleza

ADVERTENCIA
    `pDemandProfile` sale normalizado por el máximo NACIONAL, todas las zonas
    juntas. EPM espera un máximo de 1 para CADA zona. La corrección se hace en
    el paso 5 por `check_outputs.py`, y no aquí: la misma serie normalizada
    sirve también para identificar el día de punta del sistema, y hay que
    conservar entonces los tamaños relativos de las regiones.
"""

import sys
from pathlib import Path

# .absolute() y no .resolve() : resolve() remonta la junction hasta la ruta
# real, que contiene espacios ("OneDrive - WBG"). Ahora bien, el pipeline
# lanza GAMS por subprocess sin proteger las rutas con comillas: el menor
# espacio rompe el comando, GAMS no escribe Results.gdx, y el error queda
# oculto por stdout=DEVNULL. Toda la cadena debe entonces lanzarse desde
# una ruta sin espacios.
ICI = Path(__file__).absolute().parent
sys.path.insert(0, str(ICI.parent))

from representative_days.representativedays_pipeline import (  # noqa: E402
    run_representative_days_pipeline,
)

SAISONS = {m: (m - 1) // 3 + 1 for m in range(1, 13)}  # trimestres civiles
ENTREES = {
    "Load": str(ICI / "input" / "load.csv"),
    "PV": str(ICI / "input" / "pv.csv"),
    "Wind": str(ICI / "input" / "wind.csv"),
}
SORTIE = ICI / "output"
N_JOURS = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
N_CLUSTERS = 20
N_BINS = int(sys.argv[sys.argv.index("--bins") + 1]) if "--bins" in sys.argv else 10
SUFFIXE = f"_n{N_JOURS}_b{N_BINS}" if ("--n" in sys.argv or "--bins" in sys.argv) else ""
SORTIE = SORTIE.parent / (SORTIE.name + SUFFIXE)


def main():
    print("PASOS 3 Y 4 — DÍAS REPRESENTATIVOS")
    print(f"   estaciones : 4 trimestres civiles")
    print(f"   días       : {N_JOURS} optimizados por estación, más los extremos")
    print(f"   clusters   : {N_CLUSTERS}   bins : {N_BINS}")
    for k, v in ENTREES.items():
        print(f"   {k:<10} : {Path(v).name}")
    print()

    res = run_representative_days_pipeline(
        seasons_map=SAISONS,
        input_files=ENTREES,
        output_dir=SORTIE,
        epm_output_dir=SORTIE / "epm",
        summary_dir=SORTIE,
        n_representative_days=N_JOURS,
        n_clusters=N_CLUSTERS,
        n_bins=N_BINS,
        gams_main_file=str(ICI / "gams" / "OptimizationModelZone.gms"),
        verbose=True,
    )

    print()
    print("DÍAS RETENIDOS")
    rd = res.get("repr_days")
    if rd is not None:
        print(rd.to_string(index=False))
    sd = res.get("special_days")
    if sd is not None and len(sd):
        print()
        print("DÍAS EXTREMOS")
        print(sd.to_string(index=False))
    return res


if __name__ == "__main__":
    main()
