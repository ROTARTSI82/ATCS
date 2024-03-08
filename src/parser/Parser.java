package parser;

import scanner.Scanner;
import scanner.Token;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Parser.java
 * @author Grant Yang
 * @version 2024.03.06
 *
 * This parser parses the stream of tokens provided by a Scanner
 * and evaluates as it goes, interpreting Pascal on the fly.
 * It uses a right-recursive context-free grammar and parses
 * from the bottom up using recursive descent.
 */
public class Parser
{
    final private Scanner scanner;
    private Token currentToken;
    final private Map<String, Object> variables = new HashMap<>();
    private RValuePrecedenceLevel rvalueParser = null;

    private interface OperatorSAM
    {
        public abstract Object apply(Object left, Object right);
    }

    private class RValuePrecedenceLevel
    {
        private final RValuePrecedenceLevel next;
        private final Map<String, OperatorSAM> operators;

        public RValuePrecedenceLevel(Map<String, OperatorSAM> operators, RValuePrecedenceLevel next)
        {
            this.operators = operators;
            this.next = next;
        }

        public Object parse()
        {
            Object lhs = next == null ? parseFactor() : next.parse();
            while (operators.containsKey(currentToken.content()))
            {
                String op = currentToken.content();
                eat(op);
                Object rhs = next == null ? parseFactor() : next.parse();
                lhs = operators.get(op).apply(lhs, rhs);
            }
            return lhs;
        }
    }

    public Parser(Scanner scanner)
    {
        this.scanner = scanner;
        currentToken = scanner.nextToken();

        final List<Map<String, OperatorSAM>> operators = List.of(
                Map.of(
                        "*", (Object a, Object b) -> (Integer) a * (Integer) b,
                        "/", (Object a, Object b) -> (Integer) a / (Integer) b,
                        "mod", (Object a, Object b) -> (Integer) a % (Integer) b,
                        "AND", (Object a, Object b) -> (Boolean) a && (Boolean) b
                ),
                Map.of(
                        "OR", (Object a, Object b) -> (Boolean) a || (Boolean) b,
                        "+", (Object a, Object b) -> (Integer) a + (Integer) b,
                        "-", (Object a, Object b) -> (Integer) a - (Integer) b,
                        ",", (Object a, Object b) -> a.toString() + b.toString()
                ),
                Map.of(
                        "=", Object::equals,
                        "<>", (Object a, Object b) -> !a.equals(b),
                        ">=", (Object a, Object b) -> (Integer) a >= (Integer) b,
                        "<=", (Object a, Object b) -> (Integer) a <= (Integer) b,
                        ">",  (Object a, Object b) -> (Integer) a > (Integer) b,
                        "<",  (Object a, Object b) -> (Integer) a < (Integer) b
                )
        );

        for (var op: operators)
            this.rvalueParser = new RValuePrecedenceLevel(op, this.rvalueParser);
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

    private Consumer<Object> parseLValue()
    {
        String id = currentToken.content();
        eat(null, Token.Type.Identifier);

        if (currentToken.content().equals("["))
        {
            eat("[");
            int idx = (Integer) rvalueParser.parse();
            eat("]");

            return (Object o) -> ((PascalArray) variables.get(id)).set(idx, o);
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
                System.out.println(rvalueParser.parse());
                eat(")");
                eat(";");
            }
            case "READLN" ->
            {
                eat("READLN");
                eat("(");
                Consumer<Object> store = parseLValue();
                eat(")");
                eat(";");
                store.accept(new BufferedReader(new InputStreamReader(System.in)).readLine());
            }
            default ->
            {
                Consumer<Object> store = parseLValue();
                eat(":=");
                Object expr = rvalueParser.parse();
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
                Object ret = rvalueParser.parse();
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
                int lo = (Integer) rvalueParser.parse();
                eat("..");
                int hi = (Integer) rvalueParser.parse();
                eat("]");
                return new PascalArray(lo, hi);
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
                int idx = (Integer) rvalueParser.parse();
                eat("]");
                return ((PascalArray) variables.get(id)).at(idx);
            }

            return variables.get(id);
        }

        return parseNumber();
    }
}
