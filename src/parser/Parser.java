package parser;

import scanner.Scanner;
import scanner.Token;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;

public class Parser
{
    final private Scanner scanner;
    private Token currentToken;
    final private Map<String, Object> variables = new HashMap<>();

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

    private Consumer<Object> parseStoreLhs()
    {
        String id = currentToken.content();
        eat(null, Token.Type.Identifier);

        if (currentToken.content().equals("["))
        {
            eat("[");
            int idx = (Integer) parseExpr();
            eat("]");

            return (Object o) -> ((CursedArray) variables.get(id)).set(idx, o);
        }
        else
        {
            return (Object o) -> variables.put(id, o);
        }
    }

    public void parseStatement() throws IOException
    {
        switch (currentToken.content())
        {
            case "BEGIN" ->
            {
                eat("BEGIN");
                while (!currentToken.content().equals("END"))
                    parseStatement();
                eat("END");
                eat(";");
            }
            case "WRITELN" ->
            {
                eat("WRITELN");
                eat("(");
                System.out.println(parsePossibleCmp());
                eat(")");
                eat(";");
            }
            case "READLN" ->
            {
                eat("READLN");
                eat("(");
                Consumer<Object> store = parseStoreLhs();
                eat(")");
                eat(";");
                store.accept(new BufferedReader(new InputStreamReader(System.in)).readLine());
            }
            default ->
            {
                Consumer<Object> store = parseStoreLhs();
                eat(":=");
                Object expr = parsePossibleCmp();
                eat(";");
                store.accept(expr);
            }
        }
    }

    private Object parseFactor()
    {
        switch (currentToken.content())
        {
            case "(" ->
            {
                eat("(");
                Object ret = parseExpr();
                eat(")");
                return ret;
            }
            case "-" ->
            {
                eat("-");
                return -(Integer) parseFactor();
            }
            case "NOT" ->
            {
                eat("NOT");
                return !(Boolean) parseFactor();
            }
            case "TRUE" ->
            {
                eat("TRUE");
                return true;
            }
            case "FALSE" ->
            {
                eat("FALSE");
                return false;
            }
            case "array" ->
            {
                eat("array");
                eat("[");
                int lo = (Integer) parseExpr();
                eat("..");
                int hi = (Integer) parseExpr();
                eat("]");
                return new CursedArray(lo, hi);
            }
        }

        if (currentToken.type().equals(Token.Type.StringLiteral))
        {
            String ret = currentToken.content();
            eat(ret);
            return ret;
        }

        if (currentToken.type().equals(Token.Type.Identifier))
        {
            String id = currentToken.content();
            eat(id);

            if (currentToken.content().equals("["))
            {
                eat("[");
                int idx = (Integer) parseExpr();
                eat("]");
                return ((CursedArray) variables.get(id)).at(idx);
            }

            return variables.get(id);
        }

        return parseNumber();
    }

    private Object parseTerm()
    {
        Object lhs = parseFactor();
        final List<String> ops = List.of("*", "/", "AND", "mod");
        while (ops.contains(currentToken.content()))
        {
            String op = currentToken.content();
            eat(op);
            Object rhs = parseFactor();
            lhs = switch (op)
            {
                case "*" -> (Integer) lhs * (Integer) rhs;
                case "/" -> (Integer) lhs / (Integer) rhs;
                case "mod" -> (Integer) lhs % (Integer) rhs;
                default -> (Boolean) lhs && (Boolean) rhs;
            };
        }

        return lhs;
    }

    private Object parsePossibleCmp()
    {
        Object lhs = parseExpr();
        final List<String> ops = List.of("<>", ">=", "<=", "=", ">", "<");
        while (ops.contains(currentToken.content()))
        {
            String op = currentToken.content();
            eat(op);
            Object rhs = parseExpr();

            lhs = switch (op)
            {
                default -> lhs.equals(rhs);
                case "<>" -> !lhs.equals(rhs);
                case ">=" -> (Integer) lhs >= (Integer) rhs;
                case "<=" -> (Integer) lhs <= (Integer) rhs;
                case ">" -> (Integer) lhs > (Integer) rhs;
                case "<" -> (Integer) lhs < (Integer) rhs;
            };
        }
        return lhs;
    }

    private Object parseExpr()
    {
        Object lhs = parseTerm();
        final List<String> ops = List.of("+", "-", ",", "OR");
        while (ops.contains(currentToken.content()))
        {
            String op = currentToken.content();
            eat(op);
            Object rhs = parseTerm();

            lhs = switch (op)
            {
                case "+" -> (Integer) lhs + (Integer) rhs;
                case "-" -> (Integer) lhs - (Integer) rhs;
                case "," -> lhs.toString() + rhs.toString();
                default -> (Boolean) lhs || (Boolean) rhs;
            };
        }

        return lhs;
    }
}
