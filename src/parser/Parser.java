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
 * See the map definition in the constructor for a full
 * list of operators this parser supports.
 */
public class Parser
{
    final private Scanner scanner;
    private Token currentToken;
    final private Map<String, BoxedValue> variables = new HashMap<>();
    private PrecedenceLevelParser exprParser = null;

    private interface OperatorSAM
    {
        BoxedValue apply(BoxedValue left, BoxedValue right);
    }

    /**
     * This class represents a precedence level of operators to be parsed,
     * and implements a method for parsing such expressions of that precedence level
     * from the input stream. This class generalizes parseTerm() and parseExpression().
     */
    private class PrecedenceLevelParser
    {
        private final PrecedenceLevelParser next;
        private final boolean rightAssociative;
        private final Map<String, OperatorSAM> operators;

        /**
         * Construct a new parser for a specific precedence level
         * @param rtl If true, this operator is right-associative.
         * @param ops Mapping from the text of an operator to the associated action to take
         * @param next The next lower precedence level after this, to form a linked list
         */
        public PrecedenceLevelParser(boolean rtl, Map<String, OperatorSAM> ops,
                                     PrecedenceLevelParser next)
        {
            this.rightAssociative = rtl;
            this.operators = ops;
            this.next = next;
        }

        /**
         * Parse an expression of this precedence level from the input stream of
         * the parser, falling back to either the `next` precedence level
         * or to `parseFactor()` if `next` is null.
         * @return The (boxed) value of the expression in the input stream
         */
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
//                    System.out.print("operator rtl " + opName +
//                            "(" + left + ", " + right + ") = ");
                    vals.add(operators.get(opName).apply(left, right));
//                    System.out.println(vals.getLast());
                }
                else
                {
                    String opName = ops.removeFirst();
                    BoxedValue rhs = vals.removeFirst();

//                    System.out.print("operator ltr " + opName + "(" + ret + ", " + rhs + ") = ");
                    ret = operators.get(opName).apply(ret, rhs);
//                    System.out.println(ret);
                }
//                System.out.println("vals = " + vals + "\t ops = " + ops);
            }

            if (rightAssociative)
                return vals.getLast();
            else
                return ret;
        }
    }

    /**
     * Constructs a new parser to parse the tokens scanned in by
     * the scanner specified.
     * @param scanner The scanner to read from
     */
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
            this.exprParser = new PrecedenceLevelParser(rtl, op.get(rtl), this.exprParser);
        }
    }

    /**
     * Consumes a token from the input stream, with checks for its specific
     * content and type to emit an error on invalid input.
     * @param cont The expected string content of the token, or null to specify any string.
     * @param type The expected type of the token, or null to specify any type.
     * @precondition The next token in the scanner matches the expected criteria
     * @postcondition The scanner has advanced forward by 1 token
     * @throws IllegalArgumentException If the read token does not match the expected one
     */
    private void eat(String cont, Token.Type type) throws IllegalArgumentException
    {
        if ((type == null || currentToken.type().equals(type))
                && (cont == null || currentToken.content().equals(cont)))
            currentToken = scanner.nextToken();
        else
            throw new IllegalArgumentException("%s is illegal: expected %s (%s type)"
                    .formatted(currentToken, cont, type));
    }

    private void eat(String token) throws IllegalArgumentException
    {
        eat(token, null);
    }

    /**
     * Parse an integer from the input stream
     * @precondition The scanner is located at the beginning of an integer literal
     * @postcondition The scanner has advanced past the integer literal
     * @return The integer value that has appeared
     */
    private BoxedValue parseNumber()
    {
        String cont = currentToken.content();
        eat(null, Token.Type.Numeric);
        return new BoxedValue(Integer.parseInt(cont));
    }

    /**
     * Parses a and executes a Pascal statement from the input stream.
     * Statement types include blocks (BEGIN ... END;), WRITELN() calls,
     * READLN() calls, and expressions handled by `exprParser`.
     * @precondition The input stream is located at the beginning of a statement
     * @postcondition The input stream has advanced past the end of the statement,
     *                and any side effects like WRITELN and READLN have been performed.
     * @throws IOException If reading from standard input for a READLN fails
     */
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

    /**
     * Parses a factor, the highest precedence level and the basic building blocks
     * in Pascal. Factors include identifiers (variables), strings, unary operators (-, NOT),
     * parenthesis, and special values like arrays (array[1..5]) along with TRUE and FALSE
     * @precondition The input stream is located at the beginning of a valid factor expression.
     * @postcondition The input stream has advanced past the factor.
     * @return The value of the factor that appears in the input stream
     */
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
