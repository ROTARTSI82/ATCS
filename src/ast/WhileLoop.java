package ast;

public class WhileLoop implements Statement
{
    private final Expression cond;
    private final Statement body;

    public WhileLoop(Expression cond, Statement body)
    {
        this.cond = cond;
        this.body = body;
    }

    @Override
    public void exec(Environment env)
    {
        while (cond.eval(env).asBool())
            try
            {
                body.exec(env);
            }
            catch (BreakException b)
            {
                return;
            }
            catch (ContinueException c)
            {
                continue;
            }
    }
}
