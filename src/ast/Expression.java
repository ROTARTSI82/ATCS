package ast;

import parser.BoxedValue;

import static parser.BoxedValue.box;

public interface Expression
{
    public abstract BoxedValue eval(Environment env);

    Expression TRUE = (e) -> box(true);
    Expression FALSE = (e) -> box(false);
}
