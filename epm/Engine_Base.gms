$onecho > cplex.opt 
barcrossalg 1 
solutiontype 2 
lpmethod 4  
threads 12 
Names yes 
barepcomp 1e-5 
barcolnz 700 
$offEcho
$if 'x%gams.restart%'=='x' $call gams Engine_Base.gms r=Engine_Base lo=3 o=Engine_Base.lst