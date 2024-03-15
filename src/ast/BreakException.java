package ast;

public class BreakException extends RuntimeException
{
    public BreakException()
    {
        super("BREAK must be in a FOR or WHILE loop");
    }
}
