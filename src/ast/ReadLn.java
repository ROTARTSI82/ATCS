package ast;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 *
 */
public class ReadLn implements Statement
{
    private final Expression expr;
    public ReadLn(Expression targ)
    {
        expr = targ;
    }

    @Override
    public void exec(Environment env)
    {
        try
        {
            expr.eval(env).set(new BufferedReader(
                    new InputStreamReader(System.in)).readLine());
        }
        catch (Exception err)
        {
            err.printStackTrace();
        }
    }
}
