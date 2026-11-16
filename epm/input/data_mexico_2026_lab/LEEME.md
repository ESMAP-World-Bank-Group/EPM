# data_mexico_2026_lab — versión ligera, para pruebas de sensibilidad

Esta carpeta es una copia reducida de `data_mexico_2026`. Sirve para un solo
propósito: **poder correr el modelo en unos minutos en una computadora de
16 GB**, para que cada quien pueda mover un supuesto y ver el resultado dentro
de la misma sesión.

No reemplaza al modelo de referencia. Los resultados oficiales salen de
`data_mexico_2026`; los de aquí sirven para entender **sentidos y órdenes de
magnitud**, no para citar cifras.

---

## Qué se redujo

| | referencia | lab |
|---|---|---|
| Zonas | 9 | **9 — sin cambio** |
| Años | 7 (2025, 2027, 2030, 2035, 2040, 2045, 2050) | 4 (2025, 2030, 2040, 2050) |
| Días representativos | 24 (6 por temporada) | 16 (4 por temporada) |
| Centrales | 788 | 278 |
| Tamaño del problema | 10,7 M de ecuaciones | ~1,5 M |

Las **9 zonas se conservan intactas**, con sus corredores y sus límites de
transmisión. Eso es deliberado: una prueba de precio del gas no significa nada
sin la desagregación regional, porque el efecto depende de dónde está el gas y
de qué tan bien conectada está cada región.

### Los años

Se conservan 2025 (año base), 2030 (año de la meta de energías limpias), 2040 y
2050. Un evento que ocurría en un año intermedio se corre al siguiente año del
modelo: una central comprometida para 2028 entra en 2030. El modelo por tanto
**difiere un poco la entrada de las centrales comprometidas**, nunca la
adelanta.

### Los días

De los 6 días de cada temporada se conservan los 3 extremos tal cual
—`max_Load`, `min_PV`, `min_Wind`— y los 3 días representativos se fusionan en
uno solo, que recibe la totalidad de sus horas. Los 3 extremos se guardan
porque son los que cargan la punta de cada zona y el hueco renovable: sin
ellos el modelo subestima la capacidad firme que hace falta.

La suma de las horas sigue siendo **8 760** y las horas por temporada no
cambian.

### Las centrales

Las 693 centrales existentes y comprometidas se agrupan por
`zona × tecnología × combustible × estatus`, lo que da 181 agregados. Los 95
candidatos de inversión y las 2 centrales CSP **no se tocan**: los candidatos
son justamente lo que el modelo decide construir, y es ahí donde se ven los
efectos de una sensibilidad.

Cada agregado suma las capacidades y promedia el resto **ponderando por
capacidad**: consumo específico, costos fijos y variables, disponibilidad,
generación mínima obligada. La correspondencia completa entre nombres antiguos
y agregados está en `supply/_correspondance_agregation.csv`.

---

## Qué se conserva de forma exacta

Veinte controles, todos verificados:

- La capacidad instalada total y **por zona × tecnología × combustible**, al MW.
- La capacidad en línea en cada año del modelo (2025: 104 672 MW; 2030: 118 310;
  2040: 124 592; 2050: 124 265).
- Las 8 760 horas, y las horas de cada temporada.
- El **nivel medio ponderado de la demanda** y **la punta de cada una de las
  9 zonas**, sin desviación.
- El **factor de carga anual** de cada perfil renovable, por zona × tecnología
  y por agregado.
- La **disponibilidad efectiva** de cada agregado, ponderada por capacidad.

---

## Qué se pierde

1. **Los nombres de las centrales existentes.** Ya no se puede preguntar por
   Laguna Verde o por Cerro Prieto: están dentro de un agregado. Los
   candidatos sí conservan su nombre.
2. **La heterogeneidad dentro de cada agregado.** Dos ciclos combinados de la
   misma zona con consumos específicos distintos se vuelven uno con el consumo
   promedio. El modelo ya no puede despachar primero al más eficiente, así que
   **el costo de combustible queda ligeramente sobreestimado** en los años de
   demanda baja.
3. **La resolución temporal.** Con 4 años en lugar de 7, las trayectorias son
   más gruesas y la interpolación entre 2030 y 2040 es más burda.
4. Tres centrales de 160 MW en total (0,13 %) no tienen disponibilidad
   declarada. **Esto ya ocurre en el modelo de referencia**, no es algo que
   introduzca esta versión.

---

## Cómo correrlo

Desde la carpeta `epm/`:

```
python -u epm.py --folder_input data_mexico_2026_lab
```

También aparece en la lista de carpetas del tablero (dashboard).

## Una prueba de sensibilidad, paso a paso

Ejemplo: **¿qué pasa si el gas cuesta 50 % más?**

1. Copiar la carpeta `data_mexico_2026_lab` con otro nombre, por ejemplo
   `data_mexico_2026_lab_gas150`. Nunca se edita la carpeta original.
2. Abrir `supply/pFuelPrice.csv` y multiplicar por 1,5 las líneas del
   combustible `Gas`. El archivo tiene una línea por país × combustible × año.
3. Correr, y comparar con la corrida base. Las tres cifras que vale la pena
   mirar primero:
   - `pSummary`, atributo `NetPresentCost`: cuánto le cuesta al sistema;
   - `pPlantMerged`, atributo `CapacityPlant`: qué construye en lugar del gas;
   - `pEnergyBalance`: de dónde sale la energía que ya no da el gas.
4. La pregunta interesante **no es cuánto sube el costo, sino dónde**. Con las
   9 zonas intactas se puede ver qué región aguanta y qué región no, y eso
   depende de sus corredores de transmisión.

EPM también tiene un mecanismo propio de sensibilidades, declarado en
`sensitivity.csv`. Para un taller es más claro copiar la carpeta y editar el
archivo a mano: así se ve exactamente qué número cambió.
