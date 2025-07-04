* Coded: Adam Suski 10.06.2022
* Based on: Poncelet et al. - 2017 - Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning Problems
* Description: MILP optimization apprach to select typical days from the multivariate hourly time series

* Version with multi-zonal input

$ifThen not set settings
$set settings "settings.csv"
$endIf

SETS
   b         'Set of bins' 
   c         'Set of duration curves '
   d         'Set of potential representative days '
   m         'Set of medium-term periods '
   p         'Set of original time series'
   t         'Set of time steps'
   s         'Set of seasons'

;

PARAMETERS
TS(s<,d<,t<,c<)      'Original values of the parameters'
A(c,b,s,d)        'Share of the time of day d during which the lowest value of the range corresponding to bin b of duration curve c is exceeded'
L(c,b,s)          'Share of the time during which the values of a time series with corresponding duration curve c exceed the lowest value of the range corresponding to bin b'
*N                 'Number of representative periods to select'
Bins(c,b,s)       'Values of the bins for different time series'
d_s(s)            'Number of days within the season'
test
;


$if not set N $set N 3

SCALAR
N                 'Number of representative periods to select'
;
N = %N%;


VARIABLES
error(c,b,s)      'Error in approximating duration curve c at the bottom of bin b'
objective
;

POSITIVE VARIABLES
error1(c,b,s)      'Auxillary error variable to linerize absolute value'
error2(c,b,s)      'Auxillary error variable to linerize absolute value'
;

INTEGER VARIABLES
w(s,d)            'Weight assigned to day d, i.e., the number of times the representative period is assumed to be repeated within a single year'
;


BINARY VARIABLES
u(s,d)     'Binary selection variable of day d'
;


*    indexSubstitutions: {.nan: ""}
*    valueSubstitutions: {0: .nan} 
$onEmbeddedCode Connect:

- CSVReader:
    file: %settings%
    name: b
    indexColumns: [1]
    type: set


- CSVReader:
    file: %data%
    name: TS
    indexColumns: [1, 2, 3]
    header: [1]
    indexSubstitutions: {.nan: ""}
    valueSubstitutions: {0: .nan}
    trace: 2
    type: par

- GDXWriter:
    file: input.gdx
    symbols: all
    
$offEmbeddedCode


$gdxIn input.gdx
$load TS,b
$gdxIn
$offMulti


alias(d,dd);

set
s_iter(s)
;



* Lower bound for the weights
w.LO(s,d)=0;


* Calculate the number of days within the season
d_s(s) = sum(d$(sum((c,t), TS(s,d,t,c))),1);

* Calculation of the bins values
Bins(c,b,s) = smin((d,t),TS(s,d,t,c)) + ((smax((d,t),TS(s,d,t,c)) - smin((d,t),TS(s,d,t,c)))/(CARD(b)+1))*ORD(b);

* Calculating L and A parameter according to the paper
L(c,b,s) = sum((d,t)$(TS(s,d,t,c)>Bins(c,b,s)),1)/(d_s(s)*CARD(t));
A(c,b,s,d) = sum((t)$(TS(s,d,t,c)>Bins(c,b,s)),1)/(CARD(t));

Equation

EQ_OBJ              
EQ_ErrorVarDef(c,b,s)   
EQ_ErrorDef(c,b,s)   
EQ_NumDays(s)        
EQ_WeightsDef(s,d)
EQ_WeightsLim(s)
;

EQ_OBJ.. objective =E= sum((c,b,s_iter), error(c,b,s_iter));

EQ_ErrorVarDef(c,b,s_iter).. error1(c,b,s_iter) + error2(c,b,s_iter) =E= error(c,b,s_iter);

EQ_ErrorDef(c,b,s_iter).. error1(c,b,s_iter) - error2(c,b,s_iter) =E= (L(c,b,s_iter) - sum(d, w(s_iter,d)/d_s(s_iter)*A(c,b,s_iter,d)));

EQ_NumDays(s_iter)..         sum(d, u(s_iter,d)) =E= N;

EQ_WeightsDef(s_iter,d).. w(s_iter,d) =L= u(s_iter,d)*d_s(s_iter);

EQ_WeightsLim(s_iter).. sum(d, w(s_iter,d)) =E= d_s(s_iter);


Model RepresentativeDays /all/ ;
RepresentativeDays.optfile=1;

* Lopping over the seasons and solving the model for each season separately. 
loop(s,
s_iter(s) = YES;
Solve RepresentativeDays using MIP minimizing objective;
s_iter(s) = NO;
);

execute_unload 'Results' Bins, TS, L, A, w.L, u.L;
