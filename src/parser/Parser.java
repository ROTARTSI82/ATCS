package parser;

import scanner.Scanner;
import scanner.Token;

import javax.swing.*;
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
    final private Map<String, BoxedValue> variables = new HashMap<>();
    private RValuePrecedenceLevel rvalueParser = null;

    private interface OperatorSAM
    {
        public abstract Object apply(BoxedValue left, BoxedValue right);
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

        public BoxedValue parse()
        {
            BoxedValue lhs = next == null ? parseFactor() : next.parse();
            while (operators.containsKey(currentToken.content()))
            {
                String op = currentToken.content();
                eat(op);
                BoxedValue rhs = next == null ? parseFactor() : next.parse();

                // this might start to cause issues with operator=
                lhs.set(operators.get(op).apply(lhs, rhs));
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
                        "*", (a, b) -> a.asInt() * b.asInt(),
                        "/", (a, b) -> a.asInt() / b.asInt(),
                        "mod", (a, b) -> a.asInt() % b.asInt(),
                        "AND", (a, b) -> a.asBool() && b.asBool()
                ),
                Map.of(
                        "OR", (a, b) -> a.asBool() || b.asBool(),
                        "+", (a, b) -> a.asInt() + b.asInt(),
                        "-", (a, b) -> a.asInt() - b.asInt(),
                        ",", (a, b) -> a.toString() + b.toString()
                ),
                Map.of(
                        "=", (a, b) -> a.get().equals(b.get()),
                        "<>", (a, b) -> !a.get().equals(b.get()),
                        ">=", (a, b) -> a.asInt() >= b.asInt(),
                        "<=", (a, b) -> a.asInt() <= b.asInt(),
                        ">",  (a, b) -> a.asInt() > b.asInt(),
                        "<",  (a, b) -> a.asInt() < b.asInt()
                ),
                Map.of(
                        ":=", (a, b) -> a.set(b.get())
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
                System.out.println(rvalueParser.parse());
                eat(")");
                eat(";");
            }
            case "READLN" ->
            {
                eat("READLN");
                eat("(");
                BoxedValue value = rvalueParser.parse();
                eat(")");
                eat(";");
                value.set(new BufferedReader(new InputStreamReader(System.in)).readLine());
            }
            default ->
            {
                rvalueParser.parse();
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
                Object ret = rvalueParser.parse();
                eat(")");
                return new BoxedValue(ret);
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
                int lo = (Integer) rvalueParser.parse().get();
                eat("..");
                int hi = (Integer) rvalueParser.parse().get();
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
                int idx = (Integer) rvalueParser.parse().get();
                eat("]");
                return ((PascalArray) variables.get(id).get()).at(idx);
            }

            return variables.get(id);
        }

        return parseNumber();
    }
}
