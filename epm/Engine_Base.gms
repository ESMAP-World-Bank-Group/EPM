$onecho > cplex.opt 
startalg 4  
scaind  1  
lpmethod 4  
threads 4 
predual -1  
$offEcho 
$if 'x%gams.restart%'=='x' $call gams Engine_Base.gms r=Engine_Base lo=3 o=Engine_Base.lst