# GAMS (General Algebraic Modeling System) help

## Compilation vs. Execution Time in GAMS

In GAMS, compilation time refers to when the model is being processed and checked for syntax errors before execution. During this phase, GAMS interprets $ directives, defines sets, parameters, and equations but does not execute calculations or solve the model. Execution time, on the other hand, is when the model actually runs, evaluates expressions, processes loops, and solves optimization problems. The key difference is that $ directives (e.g., $if, $include) are handled at compilation time, while assignments and calculations occur during execution.

## Conditional Directives: $if, $ifi, $ifthen, $ifthen.variable
	•	$if is evaluated at compilation time and is used for simple conditional execution of code blocks. It does not allow nested conditions.
	•	$ifi is an “inline” version of $if, used when the condition should be evaluated within a single line.
	•	$ifthen / $else / $endif provides a structured way to handle multi-line conditions.
	•	$ifthen.variable is used when the condition involves a runtime variable (i.e., a GAMS parameter or scalar), making it useful for execution-time decision-making.

Example:
```
$setglobal mode test
$if "%mode%" == "test" $include test_model.gms

Scalar x /5/;
$ifthen.variable x > 4
    Display "x is greater than 4";
$endif
```
## $onmulti Directive

$onmulti allows the redefinition of sets and parameters without causing an error. Normally, GAMS prevents duplicate definitions, but this directive is useful when data may be defined multiple times, such as when reading external files or updating sets dynamically.

Example:
```
$onmulti
Set A /a, b, c/;
Set A /d, e/;  * This would normally cause an error without $onmulti
$offmulti
```
This directive should be used with caution to avoid unintended overwrites.
