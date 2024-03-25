package ast;

import parser.BoxedValue;

import static ast.NamedExpression.namedOp;
import static parser.BoxedValue.box;
import static parser.BoxedValue.newNamed;

public interface Expression
{
    BoxedValue eval(Environment env);

    Expression TRUE = namedOp((e) -> box(true), "true");
    Expression FALSE = namedOp((e) -> box(false), "false");
}
