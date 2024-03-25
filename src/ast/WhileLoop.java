package ast;

/**
 * WhileLoop.java
 * Implements a while loop, with special exception handling
 * to implement CONTINUE and BREAK
 * @version 2024.03.21
 * @author Grant Yang
 */
public class WhileLoop implements Statement
{
    private final Expression cond;
    private final Statement body;

    /**
     * Constructs a new while loop
     * @param cond The expression to check for the condition to continue looping.
     * @param body The code to execute
     */
    public WhileLoop(Expression cond, Statement body)
    {
        this.cond = cond;
        this.body = body;
    }

    /**
     * Execute the while loop, calling exec() on the body
     * statement until cond becomes false or the body throws
     * BreakException.
     * @param env Environment to execute in
     */
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
            catch (ContinueException ignored)
            {
            }
    }
}
