package parser;

import ast.*;
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
    private PrecedenceLevelParser exprParser = null;

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
        public Expression parse()
        {
            LinkedList<String> ops = new LinkedList<>();
            LinkedList<Expression> vals = new LinkedList<>();

            Expression ret = next == null ? parseFactor() : next.parse();
            if (rightAssociative)
                vals.add(ret);

            while (operators.containsKey(currentToken.content()))
            {
                String op = currentToken.content();
                eat(op);
                ops.add(op);
                vals.add(next == null ? parseFactor() : next.parse());
            }

            while (!ops.isEmpty())
            {
                if (rightAssociative)
                {
                    Expression right = vals.removeLast();
                    Expression left = vals.removeLast();
                    String opName = ops.removeLast();
                    vals.add(new BinOp(operators.get(opName), left, right));
                }
                else
                {
                    String opName = ops.removeFirst();
                    Expression rhs = vals.removeFirst();
                    ret = new BinOp(operators.get(opName), ret, rhs);
                }
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
                        "^", OperatorSAM.POW
                )),
                Map.of(false, Map.of(
                        "*", OperatorSAM.MUL,
                        "/", OperatorSAM.DIV,
                        "mod", OperatorSAM.MOD,
                        "AND", OperatorSAM.AND
                )),
                Map.of(false, Map.of(
                        "OR", OperatorSAM.OR,
                        "+", OperatorSAM.ADD,
                        "-", OperatorSAM.SUB,
                        ",", OperatorSAM.CONCAT
                )),
                Map.of(false, Map.of(
                        "=", OperatorSAM.EQ,
                        "<>", OperatorSAM.NE,
                        ">=", OperatorSAM.GE,
                        "<=", OperatorSAM.LE,
                        ">",  OperatorSAM.GT,
                        "<",  OperatorSAM.LT
                )),
                Map.of(true, Map.of(
                        ":=", OperatorSAM.SET
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
    private Expression parseNumber()
    {
        String cont = currentToken.content();
        eat(null, Token.Type.Numeric);
        return (e) -> box(Integer.parseInt(cont));
    }

    /**
     * Parses a and executes a Pascal statement from the input stream.
     * Statement types include blocks (BEGIN ... END;), WRITELN() calls,
     * READLN() calls, and expressions handled by `exprParser`.
     * @precondition The input stream is located at the beginning of a statement
     * @postcondition The input stream has advanced past the end of the statement,
     *                and any side effects like WRITELN and READLN have been performed.
     */
    public Statement parseStatement()
    {
        switch (currentToken.content())
        {
            case "WHILE" ->
            {
                eat("WHILE");
                Expression condition = exprParser.parse();
                eat("DO");
                Statement state = parseStatement();
                return new WhileLoop(condition, state);
            }
            case "IF" ->
            {
                eat("IF");
                Expression cond = exprParser.parse();
                eat("THEN");
                Statement state = parseStatement();
                return (e) ->
                { if (cond.eval(e).asBool()) state.exec(e); };
            }
            case "CONTINUE" ->
            {
                eat("CONTINUE");
                eat(";");
                return (e) ->
                { throw new ContinueException(); };
            }
            case "BREAK" ->
            {
                eat("BREAK");
                eat(";");
                return (e) ->
                { throw new BreakException(); };
            }
            case "FOR" ->
            {
                eat("FOR");
                Expression start = exprParser.parse();
                eat("TO");
                Expression end = exprParser.parse();
                eat("DO");
                Statement body = parseStatement();
                return new ForLoop(start, end, body);
            }
            case "BEGIN" ->
            {
                eat("BEGIN");
                Block blk = new Block();
                while (!currentToken.content().equals("END"))
                    blk.add(parseStatement());
                eat("END");
                eat(";");
                return blk;
            }
            case "WRITELN" ->
            {
                eat("WRITELN");
                eat("(");
                Expression value = exprParser.parse();
                eat(")");
                eat(";");
                return (e) -> System.out.println(value.eval(e).get().toString());
            }
            case "READLN" ->
            {
                eat("READLN");
                eat("(");
                Expression value = exprParser.parse();
                eat(")");
                eat(";");
                return new ReadLn(value);
            }
            default ->
            {
                Expression value = exprParser.parse();
                eat(";");
                return value::eval;
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
    private Expression parseFactor()
    {
        switch (currentToken.content())
        {
            case "(" ->
            {
                eat("(");
                Expression ret = exprParser.parse();
                eat(")");
                return ret;
            }
            case "-" ->
            {
                eat("-");
                Expression expr = parseFactor();
                return (e) -> box(-expr.eval(e).asInt());
            }
            case "NOT" ->
            {
                eat("NOT");
                Expression expr = parseFactor();
                return (e) -> box(!expr.eval(e).asBool());
            }
            case "TRUE" ->
            {
                eat("TRUE");
                return Expression.TRUE;
            }
            case "FALSE" ->
            {
                eat("FALSE");
                return Expression.FALSE;
            }
            case "array" ->
            {
                eat("array");
                eat("[");
                Expression lo = exprParser.parse();
                eat("..");
                Expression hi = exprParser.parse();
                eat("]");
                return new ArrayLiteral(lo, hi);
            }
        }

        if (currentToken.type().equals(Token.Type.StringLiteral))
        {
            String ret = currentToken.content();
            eat(ret);
            return (e) -> box(ret);
        }

        if (currentToken.type().equals(Token.Type.Identifier))
        {
            String id = currentToken.content();
            eat(id);

            if (currentToken.content().equals("["))
            {
                eat("[");
                Expression idx = exprParser.parse();
                eat("]");
                return (e) -> ((PascalArray) e.getVariable(id).get()).at(idx.eval(e).asInt());
            }

            return (e) -> e.getVariable(id);
        }

        return parseNumber();
    }
}
