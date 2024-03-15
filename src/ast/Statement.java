package ast;

public interface Statement
{
    public abstract void exec(Environment env) throws ContinueException, BreakException;
}
