package ast;

import parser.BoxedValue;

public class ForLoop implements Statement
{
    private final Expression start, stop;
    private final Statement body;

    public ForLoop(Expression start, Expression stop, Statement body)
    {
        this.start = start;
        this.stop = stop;
        this.body = body;
    }

    @Override
    public void exec(Environment env) throws ContinueException, BreakException
    {
        BoxedValue val = start.eval(env);
        BoxedValue max = stop.eval(env);
        while (val.asInt() < max.asInt())
            try
            {
                body.exec(env);
            }
            catch (BreakException b)
            {
                return;
            }
            catch (ContinueException ignored)
            {
            }
            finally
            {
                // this will execute in the case of BREAK too
                val.set(val.asInt() + 1);
            }
    }
}
