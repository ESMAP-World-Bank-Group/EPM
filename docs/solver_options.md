# Solver options
Solving an instance of the EPM model requires solving a mixed-integer programming instance with thousands of variables and constraints. The solver options chosen will then be determining to reduce computational time.

Important parameters include:
- **lpmethod**: determining the algorithm used to solve the LP problem. Options include primal simplex and barrier algorithm.
- **threads**: number of parallel threads allowed. This parameter can speed up the performance of the algorithm. The speedup is far from linear, there is usually a sweet spot between 8 and 12.
- **predual**: Switch to give dual problem to the optimizer. The setting for this parameter can either be a big winner or big looser or no impact if default makes the “right decision”.
- **barcolnz**: dense column handling. Determines whether or not columns are considered dense for special barrier algorithm handling. At the default setting of 0, this parameter is determined dynamically. Values above 0 specify the number of entries in columns to be considered as dense. This parameter can have a huge impact, but is usually difficult to tune, and is specific to each model instance.

**Note**: this list of parameters is far from exhaustive. Detailed information on how to choose the solver options are discussed [here](https://worldbankgroup.sharepoint.com/:b:/t/PowerSystemPlanning-WBGroup/EU2NwUyeOo9CljzcBCJThbsBac_sVZWv7GWmuUWf0XDIyw?e=wLkYhH).