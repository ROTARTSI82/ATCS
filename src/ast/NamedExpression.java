package ast;

import parser.BoxedValue;

/**
 * Thin wrapper around Expressions that give names to them
 */
public class NamedExpression implements Expression
{
    private Expression op;
    private String name;

    public static NamedExpression namedOp(Expression sm, String name)
    {
        NamedExpression ret = new NamedExpression();
        ret.op = sm;
        ret.name = name;
        return ret;
    }

    @Override
    public String toString()
    {
        return name;
    }

    @Override
    public BoxedValue eval(Environment env)
    {
        return op.eval(env);
    }
}
