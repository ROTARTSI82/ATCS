package ast;

public class ContinueException extends RuntimeException
{
    public ContinueException()
    {
        super("CONTINUE must be in a FOR or WHILE loop");
    }
}
