package ast;

import parser.BoxedValue;

public class BinOp implements Expression
{
    private Expression lhs, rhs;
    private String name;

    public BinOp(String type, Expression a, Expression b)
    {
        this.name = type;
        lhs = a;
        rhs = b;
    }

    @Override
    public BoxedValue eval(Environment env)
    {
        BoxedValue ret = OperatorSAM.NAME_MAP.get(name).apply(env, lhs, rhs);
        System.out.println(this + " = " + ret);
        return ret;
    }

    @Override
    public String toString()
    {
        return "(" + lhs + " " + name + " " + rhs + ")";
    }
}
