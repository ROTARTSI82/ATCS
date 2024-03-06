package parser;

import scanner.Scanner;
import scanner.Token;

public class Parser
{
    private Scanner scanner;
    private Token currentToken;

    public Parser(Scanner scanner)
    {
        this.scanner = scanner;
        currentToken = scanner.nextToken();
    }

    private void eat(String token) throws IllegalArgumentException
    {
        eat(token, null);
    }

    private void eat(String cont, Token.Type type) throws IllegalArgumentException
    {
        if ((type == null || currentToken.type().equals(type))
                && (cont == null || currentToken.content().equals(cont)))
            currentToken = scanner.nextToken();
        else
            throw new IllegalArgumentException("%s is illegal: expected %s (%s type)"
                    .formatted(currentToken, cont, type));
    }

    private int parseNumber()
    {
        String cont = currentToken.content();
        eat(null, Token.Type.Numeric);
        return Integer.parseInt(cont);
    }

    public void parseStatement()
    {
        eat("WRITELN");
        eat("(");
        System.out.println(parseFactor());
        eat(")");
        eat(";");
    }

    private int parseFactor()
    {
        if (currentToken.content().equals("("))
        {
            eat("(");
            int ret = parseFactor();
            eat(")");
            return ret;
        }
        else if (currentToken.content().equals("-"))
        {
            eat("-");
            return parseFactor();
        }
        return parseNumber();
    }
}
