package ast;

import parser.BoxedValue;
import parser.PascalArray;

import static parser.BoxedValue.box;

public class ArrayLiteral implements Expression
{
    private final Expression lowerBound, upperBound;

    public ArrayLiteral(Expression lo, Expression hi)
    {
        lowerBound = lo;
        upperBound = hi;
    }

    @Override
    public BoxedValue eval(Environment env)
    {
        return box(new PascalArray(lowerBound.eval(env).asInt(), upperBound.eval(env).asInt()));
    }
}
