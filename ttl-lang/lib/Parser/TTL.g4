grammar TTL;

module:         function+ ;

function:       'function' type ID argumentList '{' statement* '}' ;

argumentList:   '(' argument (','argument)* ')'
                | '(' ')'
                ;

argument:       type ID ;

statement:      varDef
                | varAssign
                | callStmt
                | forLoop
                | ifStmt
                | returnStmt
                | compoundStmt
                ;

varDef:         'var' type ID ('=' expr )? ';' ;
varAssign:      lhsID '=' expr ';' ;
lhsID:          ID '[' expr (',' expr)* ']'                                       # MatrixElemAssign
                | ID                                                              # PrimAssign
                ;
callStmt:       callExpr ';' ;
forLoop:        'for' '(' ID 'in' expr 'by' expr ')' statement ;
ifStmt:         'if' '(' expr ')' statement ('else' statement)?;
returnStmt:     'return' expr ';' ;
compoundStmt:   '{' statement* '}' ;

type:           scalarType
                | voidType
                | matrixType
                ;

scalarType:		  intType | floatType;
intType:        'int';
floatType:      'float';
voidType:       'void' ;
matrixType:     'matrix' '<' scalarType '>' '[' dim (',' dim)* ']' ;

dim:            INT | '?' ;

expr:
// Matrix expressions
                expr '{' expr (',' expr)* '}'                                     # SliceMatrix
                | expr '#' expr                                                   # MatrixMul
                | expr '$' expr                                                   # Dimension
// Arithmetic expressions
                | '-' expr                                                        # UnaryMinus
                | expr ('*'|'/') expr                                             # Multiplication
                | expr ('+'|'-') expr                                             # Addition

// Boolean expressions
                | expr ('>'|'<'|'<='|'>='|'=='|'!=') expr                         # Compare
                | '!' expr                                                        # BooleanNot
                | expr '&' expr                                                   # And
                | expr '|' expr                                                   # Or
// Atoms
                | '[' expr (',' expr)* ']'                                        # MInit
                | '(' expr ')'                                                    # ParExpr
                | expr '...' expr                                                 # Range
                | callExpr                                                        # Call
                | ID                                                              # IdRef
                | INT                                                             # IntAtom
                | FLOAT                                                           # FloatAtom
                | ':'                                                             # WholeRange
                ;

callExpr:       ID parameterList ;
parameterList:  '(' expr (',' expr)* ')'
                | '(' ')' ;

// Lexer rules
ID:             ([a-z]|[A-Z]) ([a-z]|[A-Z]|NUM|'_')* ;

FLOAT:          NUM '.' [0-9]+ ;
INT:            NUM  ;
fragment NUM :  [0-9] [0-9]* ;

LINE_COMMENT:    '//' .*? '\r'? '\n' -> skip ;
COMMENT:        '/*' .*? '*/' -> skip ;
WS:             [ \t\r\n]+ -> skip ;
