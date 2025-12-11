"""An implementation of a Pratt Parser for simple expressions"""
from levycas.expressions import *
from levycas import simplify

import re

class Token:
    """A token object is a substring + metadata
    """

    def __init__(self, token_type: str, token_value: str, token_pos: int) -> None:
        """Create a new token object.

        Args:
            token_type (str): The type of the matched substring.
            token_value (str): The matched substring in the expression.
            pos (int): The position of the matched substring in the expression.
        """
        self.type = token_type
        self.value = token_value
        self.pos = token_pos
        self.len = len(token_value)

    def __repr__(self) -> str:
        return f"<Token:: {self.type} [{self.pos}-{self.pos + self.len}]>"

def tokenize(expr: str, token_spec: list) -> list[Token]:
    """Tokenize an expression string

    Args:
        expr (str): The input expression to tokenize
        token_spec (list): The token specifications
    """
    try:
        token_pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
    except:
        raise SyntaxError("Incorrect token specification")

    tokens = list()
    for match in re.finditer(token_pattern, expr):
        token_type = match.lastgroup
        token_value = match.group()
        token_pos = match.span()[0]

        if token_type == "SPACE":
            continue
        
        elif token_type == "OTHER":
            raise SyntaxError(f"Invalid token: {token_value} at position {match.span()[0]}")

        else: 
            tokens.append(Token(token_type, token_value, token_pos))
    return tokens

#Global verbose vars for testing purposes
pv = False #parsing verbose 
lv = False #lexing verbose


"""Token specifications, matching regex strings"""
TOKEN_SPEC: list[tuple[str]] = [
    #Elementary functions, case insensitive
    ("ELEM_SIN",       r'sin'),          #Matches sin function
    ("ELEM_COS",       r'cos'),          #Matches cos function
    ("ELEM_TAN",       r'tan'),          #Matches tan function
    ("ELEM_CSC",       r'csc'),          #Matches csc function
    ("ELEM_SEC",       r'sec'),          #Matches sec function
    ("ELEM_ARCTAN",    r'arctan'),       #Matches arctan function
    ("ELEM_ARCCOS",    r'arccos'),       #Matches arccos function
    ("ELEM_ARCSIN",    r'arcsin'),       #Matches arcsin function
    ("ELEM_LN",        r'ln'),           #Matches natural log
    ("ELEM_EXP",       r'exp'),          #Matches exponential

    #Constants/Symbols
    ("NUMBER",    r'(\d+\.?\d*|\d*\.\d+)'),   #Matches a decimal
    ("VARIABLE",  r'[a-zA-Z]'),               #Matches a single letter

    #Elementary operations
    ("POW",       r'\^'),                     #Matches a single power symbol
    ("PLUS",      r'\+'),                     #Matches a plus sign
    ("MINUS",     r'-'),                      #Matches a minus sign
    ("MULT",      r'\*'),                     #Matches a multiplication sign
    ("DIV",       r'/'),                      #Matches a division sign
    ("FACT",      r'\!'),                     #Matches a factorial symbol

    #Special characters
    ("LPAREN",    r'\('),                     #Matches a left paren
    ("RPAREN",    r'\)'),                     #Matches a right paren
    ("SPACE",     r'\s+'),                    #Matches any whitespace
    ("OTHER",     r'.'),                      #Matches any invalid characters
    ("EOL",       r'\$')                      #Matches the End-of-Line character '$'
]

"""Token classes, matching associated Expression classes"""
TOKEN_CLASS = {
    #Constants/Symbols
    "NUMBER": Integer,
    "VARIABLE": Variable,

    #Elementary Functions
    "ELEM_SIN": Sin,
    "ELEM_COS": Cos,
    "ELEM_TAN": Tan,
    "ELEM_CSC": Csc,
    "ELEM_SEC": Sec,
    "ELEM_ARCTAN": Arctan,
    "ELEM_ARCCOS": Arccos,
    "ELEM_ARCSIN": Arcsin,
    "ELEM_LN": Ln,
    "ELEM_EXP": Exp,

    #Elementary operations, "MINUS" parsed separately
    "POW": Power,
    "PLUS": Sum,
    "MULT": Product,
    "DIV": Div,
    "FACT": Factorial,

}

"""Operator precedence rules, matching associated binding powers"""
TOKEN_BP = {
    #Constants/Symbols
    "NUMBER": (0, 0),
    "VARIABLE": (0, 40),

    #Elementary functions, case insensitive
    "ELEM_SIN": (5, 40),
    "ELEM_COS": (5, 40),
    "ELEM_TAN": (5, 40),
    "ELEM_CSC": (5, 40),
    "ELEM_SEC": (5, 40),
    "ELEM_ARCTAN": (5, 40),
    "ELEM_ARCCOS": (5, 40),
    "ELEM_ARCSIN": (5, 40),
    "ELEM_LN": (5, 40),
    "ELEM_EXP": (5, 40),

    #Elementary operations
    "POW": (35, 30),
    "PLUS": (10, 15),
    "MINUS": (10, 20),
    "MULT": (20, 25),
    "DIV": (20, 25),
    "FACT": (40, 40),

    #Special characters
    "LPAREN": (-1, -1),
    "RPAREN": (-1, -1),
    "SPACE": (-1, -1),
    "OTHER": (-1, -1),
    "EOL": (-1, -1)
}

def parse(expression: str, **symbols) -> Expression:
    """Parse an input expression into a LevyCAS Expression tree. 

    Args:
        expression (str): A mathematical expression

    Returns:
        Expression.Expression: The root of the AST representing the input expression.
    """
    tokens: list[Token] = tokenize(expression, TOKEN_SPEC)
    tokens = expand(tokens, **symbols) #Expand implicit multiplications, if present
    tokens.append(Token("EOL", "$", -1)) #Append end of line token
    tokens = tokens[::-1] #Reverse token list to pop from end
    expr = pratt(tokens, 0, **symbols)
    assert tokens.pop().type == "EOL" and len(tokens) == 0
    return expr

def expand(tokens: list[Token], **symbols) -> list[Token]:
    """Expand implicit multiplications"""
    if lv:
        print(f"Expanding with: {symbols=}")
    expanded = list()
    for i in range(len(tokens)):
        curr = tokens[i]

        #Symbols previously defined as functions are replaced
        if curr.type == "VARIABLE":
            sym = symbols.get(curr.value, Variable(-1))
            if isinstance(sym, Function):
                curr.type = "FUNCTION"

        expanded.append(curr)

        #Implicit Multiplication is expanded
        next = tokens[i + 1] if i + 1 < len(tokens) else None
        if next:
            if curr.type in ("VARIABLE", "NUMBER", "RPAREN", "DECIMAL") and (next.type in ("VARIABLE", "LPAREN", "FUNCTION") or next.type.startswith("ELEM_")):
                expanded.append(Token("MULT", "*", curr.pos + curr.len))
    return expanded

def pratt(tokens: list[Token], bp: int, **symbols) -> Expression:
    """Recursive Pratt Parsing function

    Args:
        tokens (list[Token]): List of tokens
        bp (int): Right binding power of the preceding operator

    Returns:
        Expression: The root of the subtree parsed by this iteration
    """
    if lv:
        print(f"Tokens: {tokens[::-1]}")

    curr = tokens.pop()
    if pv: 
        print(f"Parsing curr: {curr.type}: {curr.value}")

    #Parse first token as left hand side (NULL DENOTATIONS)

    if curr.type == "NUMBER":
        #Parse decimal number
        number = curr.value.split(".")
        if len(number) == 1:
            left = Integer(int(curr.value))
        else:
            assert len(number) == 2, f"Failed to parse number {curr.value}"
            whole, partial = number
            
            denominator = 10 ** len(partial)
            numerator = int(whole + partial)
            if denominator == 1:
                left =  Integer(numerator)
            else:
                left = Rational(numerator, denominator)

    elif curr.type == "VARIABLE":
        left = Variable(curr.value)

    elif curr.type.startswith("ELEM_"):
        right = pratt(tokens, TOKEN_BP[curr.type][1])
        left = TOKEN_CLASS[curr.type](right)

    elif curr.type == "LPAREN":
        left = pratt(tokens, 0, **symbols)
        next = tokens.pop()
        if next.type != "RPAREN" and next.type != "COMMA":
            raise SyntaxError(f"Expected closing parenthesis from {next}")

    elif curr.type == "PLUS":
        left = pratt(tokens, 0, **symbols) #Ignore unary plus operator

    elif curr.type == "MINUS": #Unary minus operator is parsed into (-1) * Exp
        minus_rbp = TOKEN_BP["MINUS"][1] 
        right = pratt(tokens, minus_rbp, **symbols)
        left = -right

    else:
        raise SyntaxError(f"Expected null denotation from {curr}")

    #Parse operator and recurse on remaining expression
    while True:
        next = tokens[-1]
        if pv:
            print(f"Looping {left=}, {next=}")
        if next.type == "RPAREN" or next.type == "EOL": #Check that next token is not the end of an expression
            break #Break without consuming token

        if next.type == "COMMA": #End of a function argument.
            #Note that if an extraneous comma is present, the
            #end-of-line token will not be next, raising an error.

            #break with consuming token
            if pv:
                print(f"Detected comma, returning: {left}")
            break

        lbp, rbp = TOKEN_BP[next.type]

        if lbp <= 0 or rbp <= 0: #Check that next token is an operator
            raise SyntaxError(f"Expected operator from {next}")

        if lbp < bp: #Checks L/R associativity against previous operator
            break

        tokens.pop() #Consume token after precedence check

        #Basic operations (in/post-fix)
        if next.type == "MINUS":
            right = pratt(tokens, rbp, **symbols)
            left -= right
        elif next.type == "PLUS":
            right = pratt(tokens, rbp, **symbols)
            left += right
        elif next.type == "POW":
            right = pratt(tokens, rbp, **symbols)
            left = left ** right
        elif next.type == "DIV":
            right = pratt(tokens, rbp, **symbols)
            left /= right
        elif next.type == "MULT":
            right = pratt(tokens, rbp, **symbols)
            left *= right
        elif next.type == "FACT":
            left = Factorial(left)
        else:
            print(f"{next.type} is not a supported operation?")

    return left