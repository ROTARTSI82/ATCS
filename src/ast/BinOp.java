package ast;

import parser.BoxedValue;

public class BinOp implements Expression
{

    private Expression lhs, rhs;
    private OperatorSAM type;

    public BinOp(OperatorSAM type, Expression a, Expression b)
    {
        this.type = type;
        lhs = a;
        rhs = b;
    }

    @Override
    public BoxedValue eval(Environment env)
    {
        return type.apply(env, lhs, rhs);
    }
}
