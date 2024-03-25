package ast;

/**
 * Statement.java
 * Implements a Statement, which executes in an environment and
 * causes side effects rather than returning a value like Expression.
 * @author Grant Yang
 * @version 2024.03.21
 */
public interface Statement
{
    /**
     * Execute the statement
     * @param env The environment to execute in
     * @throws ContinueException If any CONTINUE statement was reached. This is caught by loops.
     * @throws BreakException If any BREAK statement was reached. This is caught by loops
     */
    void exec(Environment env) throws ContinueException, BreakException;
}
