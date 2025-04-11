# Set definition

## Implicit set definition

Sets are declared at the beginning of the base.gms file. Example:
```
Sets
   g        'generators or technology-fuel types'
   f        'fuels'
   tech     'technologies'
   y        'years'
   q        'quarters or seasons'
   d        'day types'
   t        'hours of day'
   z        'zones'
   c        'countries'
   zext     'external zones'
*********Hydrogen specific addition***********
   hh        'Hydrogen production units'
;

```

These sets are populated in `main.gms`, either by reading them directly (e.g. `y` from `y.csv`) or implicitly through data parameters.

An implicit definition is identified by the `<` syntax in the parameter declaration:

``` 
Parameter
* Generator data
   pGenDataExcel(g<,z,tech<,f<,*)      'Generator data from Excel input'
```

This implies that the fuel set `f` is defined by all the sets that appear in the file `pGenDataExcel`

### Set definition in EPM

We discuss here where the main sets are being defined:
- q (season): defined in `pHours`
- d (days): defined in `pHours`
- t (hours): defined in `pHours`
- z (zones): defined in `zcmap`
- c (countries): defined in `zcmap`
- g (generators): defined in `pGenDataExcel`
- f (fuels): defined in `pGenDataExcel`
- tech (techs): defined in `pGenDataExcel`