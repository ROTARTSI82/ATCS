package parser;

import scanner.Scanner;
import scanner.Token;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import static parser.BoxedValue.box;

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
    final private Map<String, BoxedValue> variables = new HashMap<>();
    private PrecedenceLevel exprParser = null;

    private interface OperatorSAM
    {
        BoxedValue apply(BoxedValue left, BoxedValue right);
    }

    private class PrecedenceLevel
    {
        private final PrecedenceLevel next;
        private final boolean rightAssociative;
        private final Map<String, OperatorSAM> operators;

        public PrecedenceLevel(boolean rtl, Map<String, OperatorSAM> ops, PrecedenceLevel next)
        {
            this.rightAssociative = rtl;
            this.operators = ops;
            this.next = next;
        }

        public BoxedValue parse()
        {
            LinkedList<String> ops = new LinkedList<>();
            LinkedList<BoxedValue> vals = new LinkedList<>();

            BoxedValue ret = next == null ? parseFactor() : next.parse();
            if (rightAssociative)
                vals.add(ret);

            while (operators.containsKey(currentToken.content()))
            {
                String op = currentToken.content();
                eat(op);
                ops.add(op);
                vals.add(next == null ? parseFactor() : next.parse());
            }

//            System.out.println("layer " + operators.keySet() +
//                    ": vals = " + vals + " ops = " + ops);
            while (!ops.isEmpty())
            {
                if (rightAssociative)
                {
                    BoxedValue right = vals.removeLast();
                    BoxedValue left = vals.removeLast();
                    String opName = ops.removeLast();

                    System.out.print("operator rtl " + opName + "(" + left + ", " + right + ") = ");
                    vals.add(operators.get(opName).apply(left, right));
                    System.out.println(vals.getLast());
                }
                else
                {
                    String opName = ops.removeFirst();
                    BoxedValue rhs = vals.removeFirst();

                    System.out.print("operator ltr " + opName + "(" + ret + ", " + rhs + ") = ");
                    ret = operators.get(opName).apply(ret, rhs);
                    System.out.println(ret);
                }
//                System.out.println("vals = " + vals + "\t ops = " + ops);
            }

            if (rightAssociative)
                return vals.getLast();
            else
                return ret;
        }
    }

    public Parser(Scanner scanner)
    {
        this.scanner = scanner;
        currentToken = scanner.nextToken();

        final List<Map<Boolean, Map<String, OperatorSAM>>> operators = List.of(
                Map.of(true, Map.of(
                        "^", (a, b) -> box((int) Math.pow(a.asInt(), b.asInt()))
                )),
                Map.of(false, Map.of(
                        "*", (a, b) -> box(a.asInt() * b.asInt()),
                        "/", (a, b) -> box(a.asInt() / b.asInt()),
                        "mod", (a, b) -> box(a.asInt() % b.asInt()),
                        "AND", (a, b) -> box(a.asBool() && b.asBool())
                )),
                Map.of(false, Map.of(
                        "OR", (a, b) -> box(a.asBool() || b.asBool()),
                        "+", (a, b) -> box(a.asInt() + b.asInt()),
                        "-", (a, b) -> box(a.asInt() - b.asInt()),
                        ",", (a, b) -> box(a.get().toString() + b.get().toString())
                )),
                Map.of(false, Map.of(
                        "=", (a, b) -> box(a.get().equals(b.get())),
                        "<>", (a, b) -> box(!a.get().equals(b.get())),
                        ">=", (a, b) -> box(a.asInt() >= b.asInt()),
                        "<=", (a, b) -> box(a.asInt() <= b.asInt()),
                        ">",  (a, b) -> box(a.asInt() > b.asInt()),
                        "<",  (a, b) -> box(a.asInt() < b.asInt())
                )),
                Map.of(true, Map.of(
                        ":=", (a, b) -> a.set(b.get())
                ))
        );

        for (var op: operators)
        {
            boolean rtl = op.containsKey(true);
            this.exprParser = new PrecedenceLevel(rtl, op.get(rtl), this.exprParser);
        }
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

    private BoxedValue parseNumber()
    {
        String cont = currentToken.content();
        eat(null, Token.Type.Numeric);
        return new BoxedValue(Integer.parseInt(cont));
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
                System.out.println(exprParser.parse().get());
                eat(")");
                eat(";");
            }
            case "READLN" ->
            {
                eat("READLN");
                eat("(");
                BoxedValue value = exprParser.parse();
                eat(")");
                eat(";");
                value.set(new BufferedReader(new InputStreamReader(System.in)).readLine());
            }
            default ->
            {
                exprParser.parse();
                eat(";");
            }
        }
    }

    private BoxedValue parseFactor()
    {
        switch (currentToken.content())
        {
            case "(" ->
            {
                eat("(");
                BoxedValue ret = exprParser.parse();
                eat(")");
                return ret;
            }
            case "-" ->
            {
                eat("-");
                return new BoxedValue(-(Integer) parseFactor().get());
            }
            case "NOT" ->
            {
                eat("NOT");
                return new BoxedValue(!(Boolean) parseFactor().get());
            }
            case "TRUE" ->
            {
                eat("TRUE");
                return new BoxedValue(true);
            }
            case "FALSE" ->
            {
                eat("FALSE");
                return new BoxedValue(false);
            }
            case "array" ->
            {
                eat("array");
                eat("[");
                int lo = (Integer) exprParser.parse().get();
                eat("..");
                int hi = (Integer) exprParser.parse().get();
                eat("]");
                return new BoxedValue(new PascalArray(lo, hi));
            }
        }

        if (currentToken.type().equals(Token.Type.StringLiteral))
        {
            String ret = currentToken.content();
            eat(ret);
            return new BoxedValue(ret);
        }

        if (currentToken.type().equals(Token.Type.Identifier))
        {
            String id = currentToken.content();
            eat(id);

            if (!variables.containsKey(id))
                variables.put(id, BoxedValue.newNamed(id));

            if (currentToken.content().equals("["))
            {
                eat("[");
                int idx = (Integer) exprParser.parse().get();
                eat("]");
                return ((PascalArray) variables.get(id).get()).at(idx);
            }

            return variables.get(id);
        }

        return parseNumber();
    }
}
