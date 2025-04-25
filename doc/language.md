# Toy Tensor Language

This is the language documentation for the Toy Tensor Language or TTL for short.

As the name indicates, this language is not intended to be a full-blown
programming language, but rather a vehicle to demonstrate the basics of
building an MLIR-based compiler.

## Language Basics

The language is not sensitive to whitespace or identation.

Comments in code can be inserted with `//` for a line comment or `/*` followed
by `*/` for a block comment.

Integer numeric literals are given through a sequence of one or more numbers
0-9.

Floating-point numeric literals are given through a sequence of one or more
numbers 0-9, followed by `.` and another sequence of one or more numbers 0-9.
Scientific notation is currently not supported.

Identifiers need to start with a lower- or upper-case character and consist of
one or more characters. Identifiers can also contain numeric characters or `_`,
but not as the first character.

## Type System

There are two basic scalar types in TTL:

- `int`: A 32 bit wide, signed integer type. Any operation that leads to an
  under- or overflow is undefined behavior.
- `float`: A single-precision, IEEE-754 compliant floating point type.

The additional `void` type can only be used as return type of functions to
indicate that the function does not return a value.

In addition, the language allows the definition of matrix types with arbitrary
dimensionality.

The definition of a matrix type starts with the keyword `matrix`. Next, the
element type of the matrix is given, enclosed by `<` and `>`. The only allowed
element types are `int` and `float`. Lastly, the dimensionality and sizes of
the matrix are given by a list enclosed by `[` and `]`. The sizes can either be
statically known and specified by an integer numeric literal or be dynamic,
indicated by `?`.

Examples:
```
// An integer matrix with a single dimension and four elements.
matrix <int> [4]

// A float matrix with two dimensions and a dynamic number of elements in
// both dimensions
matrix <float> [?, ?]

// An integer matrix with two dimensions, where one is dynamic and one has
// 6 elements.
matrix <int> [?, 6]
```

Next to these explicit types, the language implictly uses a `range` type. It is
however not possible to define variables of `range` type and the use of values
of `range` type is limited to specific language constructs.

## Structure of a Program

A program consists of a single translation unit ("module"). Inclusion of
content from other files is not possible.

A module consists of one or more function definitions. Functions must be
defined before they can be used through calls in other functions.

A function with the name `main`, `void` or `int` return type and no arguments
must be defined as the entry point to the program.

## Functions

Function definitions start with the keyword `function`, followed by the
return type of the function, where all types, except for the implicit `range`
type are allowed as return type.

As next element, the name of the function must be given as identifier. Function
names must be unique per translation unit, overloading of functions is not
supported.

The name is followed by the argument list of the function, enclosed by `(` and
`)`. Each argument must be given a type, where all types except the implict
`range` type are allowed, and a name given as valid identifier. The argument
list can consist of zero or more arguments.

The last element is the body of the function, a sequence of one or more
statements enclosed by `{` and `}`.

## Statements

Each statement ends with `;`. In places where a single statement is expected 
to appear, either a single statement can be given or a sequence of statements
as compound statement can be given, where a compound statement is a sequence
of one or more statements enclosed by `{` and `}`.

### Variable Definition

All variables must be declared before use. The default scope of variables is
function scope. The scope of variable can further be limited by defining it
inside a compound statement, the compound statements starts a new scope before
its first statement and ends that scope after its last statement. Each variable
name must only be used once per scope. In case of nested scopes through
compound statements, variables declared inside the nested scope shadow
declarations from the enclosing scope(s).

A variable declaration starts with the keyword `var`, followed by the type of
the variable, where all types except for the implicit `range` type are allowed.

Following the type, the name of the variable must be given as a valid
identifier.

Optionally, the variable can be initialized at the place of declaration by
adding `=` followed by an expression with the initial value.

Variables of all types can be initialized with an expression that matches the
type of the variable exactly.

Matrix variables can additional be initialized in additional ways, if and only
if their shape is entirely static:
- Initialization of a multi-dimensional matrix from a matrix expression with
  only one dimension, if the number of elements are identical.
- Initialization of a matrix with a range. If the range is dynamic, i.e., its
  start or end are given by a variable not known at compile time, it is
  undefined behavior if the number of elements in the range does not match
  the numer of elements in the matrix at runtime. This initialization is only
  possible for matrix with integer element type.
- Initialization of a matrix with a single scalar, where the single value is
  broadcasted to all elements of the matrix. The type of the scalar must match
  the element type of the matrix.

If no initial value is given, variables of type `int` or `float` are initialized
to `0`, whereas matrix types are initialized with an unknown value.

### Variable Assignment

Variable assignment happens by specifying a valid variable identifier on the
left-hand side of `=` and an expression that matches the type of the variable
identified by the identifier on the right-hand side.

For matrix variables it is additionally possible to assign a specific element
of the matrix by specifying a range of indices enclosed by `[` and `]`. The
number of indices specified must match the dimensionality of the matrix and all
index expressions must evaluate to a single scalar of `int` type. Assigning
outside the bounds of a matrix is undefined behavior.

### Call Statement

Function calls can form a statement. If the called function returns a result,
that result is discarded in this case.

### For Loops

Loops start with the keyword `for`, followed by the definition of the loops
iteration space enclosed in `(` and `)`. The iteration space is defined by a
valid variable identifier, followed by the keyword `in`, an expression
evaluating to a `range`, the keyword `by` and an expression evaluation to a
single value of type `int`.

Following the definition of the iteration space, is a statement as body of the
loop (see compound statement to use multiple statements as the body of the
function).

Example:
```
// A loop iteration from 0 (inclusive) to 8 (exclusive) in steps of 2.
var int i = 0;
for(i in 0...8 by 2)
  foo(i);
```

### Conditional Execution

Conditional execution is enabled by the `if` statement. An `if` statement starts
with the keyword `if`, followed by a condition enclosed by `(` and `)`. The
condition must be given by an expression evaluating to a single value of type
`int`, where `0` indicates the false value and every other value indicates the
true value.

The code to execute if the condition is true is given by a statement following
the condition.

Optionally, an alternative branch of execution can be given by the keyword
`else` followed by the statement to execute in case the condition evaluates to
the false value.

### Return Statement

A return statement is given by the keyword `return` followed by an expression.
The expression must evaluate to a single value that matches the return type
of the function enclosing the return statement. Functions returning no value
(`void`) must not contain a return statement.
Return statements must only appear as the last statement in the body of a
function, i.e., an early or conditional return is not possible.

## Expressions

Every expression evaluates to a single value of a specific type, which
includes the implicit `range` type.

### Basic Expressions

Integer or floating point numeric literals are valid expressions of type `int`
and `float`, respectively.

A reference to a valid variable identifier is a valid expression and the type
of this expression matches the declared type of the identified variable.

An expression enclosed by `(` and `)` is a valid expression and the type is
given by the type of the enclosed expression.

### Initializer list

A list of expressions, enclosed by `[` and `]` and separated by `,` is an
initializer list. The types of all elements of the list must match and must
be of type `int` or `float`.

The resulting type is a `matrix` type with the element type given by the
expressions in the list. The matrix is one-dimensional, with the static size
in that single dimension given by the number of entries in the initializer list.

### Range Expression

Two expressions that each evaluate to a single value of type `int`, separated
by `...` form a range expression. The resulting type is the implicit `range`
type. The left hand expression forms the inclusive start of the range, the
right hand expression forms the exclusive end of the range.

### Call Expression

A call expression is a valid function identifier followed by a list of
argument expressions, enclosed by `(` and `)` and separated by `,`. The number
and types of the arguments must match the parameter types of the identified
function.

The resulting type of the call expression is the return type of the identified
function.

### Matrix Slice Expression

The matrix slice expression allows to extract a single element or a sub-slice
from a matrix.

It is defined by an expression of matrix type, followed by a list of indices
enclosed by `{` and `}` and separated by `,`. The indices must be expressions
of type `int` or `range` and the number of indices must match the dimensionality
of the matrix.

For the resulting matrix, the size in a dimension is dynamic (`?`) if the
index slice expression for this dimension was of type `range`, otherwise
(i.e., slice expression of type `int`) the size is `1`. In the special case that
all index slice expressions are of type `int`, the resulting type of the matrix
slice expression is not `matrix`, but a scalar element with the element type
of the original matrix.

Slices that would lead to out-of-bounds access of the matrix result in
undefined behavior.

### Matrix Multiplication

Two matrices can be multiplied by separating two expressions of matrix type with
the `#` operator.

Matrix multiplication is only possible for two-dimensional matrices. The first
matrix must be of size NxK, the second matrix must be of size KxM. Note that
this can only be enforced by the language for matrices with static sizes. In
case of dynamic sizes it is undefined behavior if the sizes do not fulfill this
requirement.

The element type of the two matrices must match.

The result is a matrix of size NxM, with the elemnent type being the element
type of the original matrices.

### Matrix Size Expression

To query the size of matrix in a specific dimension (static or dynamic size),
an expression of matrix type and an expression of integer type are separated by
`$`. The result of this expression is `int`.

It is undefined behavior if the dimension given on the right hand side of `$`
exceeds the dimensionality of the matrix.

### Arithmetic expressions

The unary minus expression `-` negates the value of a variable of type `int` or
`float`.

The binary arithmetic expressions `*` (multiplication), `/` (division),
`+` (addition) and `-` (subtraction) can be used in multiple ways:
- If both operand expressions on the left and right side are of matrix type,
  they perform an elementwise arithmetic operation. In that case, the shape and
  the element type of the two matrices must match and it is undefined behavior
  if they do not.
- If one of the operand expressions is of matrix type and the other is of scalar
  type that matches the matrix operands element type, the arithmetic operation
  is performed for each element of the matrix with the scalar operand.
- If both operands are of scalar type, the operand types must match.

### Comparison

The comparison operators `>` (greater than), `<` (less than), `<=` (less or
equal), `>=` (greater or equal), `==` (equal) and `!=` (not equal) with operand
expressions on the left and right side of the operator form a comparison
expression.

The types of both operand expressions must match and must be of type `int` or
`float`. The resulting type is always `int`.

### Logic expressions

All logic expressions can only be applied to values of type `int`.

The unary not `!` expression inverts a boolean value.

The binary operators `&` (and) and `|` (or) perform a logical "and" and "or"
operation on boolean values.
