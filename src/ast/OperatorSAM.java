package ast;

import parser.BoxedValue;

import java.util.List;
import java.util.Map;

import static parser.BoxedValue.box;

/**
 * OperatorSAM.java
 * @author Grant Yang
 * @version 2024.03.21
 * An interface for a Single Abstract Method to implement
 * any binary operator expression.
 */
public interface OperatorSAM
{
    BoxedValue apply(Environment env, Expression left, Expression right);

    /**
     * NAME_MAP maps the text of an infix operator to the OperatorSAM
     * that actually implements it.
     */
    Map<String, OperatorSAM> NAME_MAP = Map.ofEntries(
            Map.entry("^", (e, a, b) -> box((int) Math.pow(a.eval(e).asInt(), b.eval(e).asInt()))),
            Map.entry("*", (e, a, b) -> box(a.eval(e).asInt() * b.eval(e).asInt())),
            Map.entry("/", (e, a, b) -> box(a.eval(e).asInt() / b.eval(e).asInt())),
            Map.entry("mod", (e, a, b) -> box(a.eval(e).asInt() % b.eval(e).asInt())),
            Map.entry("AND", (e, a, b) -> box(a.eval(e).asBool() && b.eval(e).asBool())),
            Map.entry("OR", (e, a, b) -> box(a.eval(e).asBool() && b.eval(e).asBool())),
            Map.entry("+", (e, a, b) -> box(a.eval(e).asInt() + b.eval(e).asInt())),
            Map.entry("-", (e, a, b) -> box(a.eval(e).asInt() - b.eval(e).asInt())),
            Map.entry(",", (e, a, b) -> box(a.eval(e).get().toString() + b.eval(e).get().toString())),
            Map.entry("=", (e, a, b) -> box(a.eval(e).get().equals(b.eval(e).get()))),
            Map.entry("<>", (e, a, b) -> box(!a.eval(e).get().equals(b.eval(e).get()))),
            Map.entry(">=", (e, a, b) -> box(a.eval(e).asInt() >= b.eval(e).asInt())),
            Map.entry("<=", (e, a, b) -> box(a.eval(e).asInt() <= b.eval(e).asInt())),
            Map.entry(">", (e, a, b) -> box(a.eval(e).asInt() > b.eval(e).asInt())),
            Map.entry("<", (e, a, b) -> box(a.eval(e).asInt() < b.eval(e).asInt())),
            Map.entry(":=", (e, a, b) ->
            {
                Object v = b.eval(e).get();
                return a.eval(e).set(v);
            })
    );

    /**
     * PRECEDENCE contains a list of operators with the same precedence,
     * with the highest precedence operators coming first.
     * The boolean value in the map entry specifies whether the operators
     * should be left or right associative, with a value of TRUE denoting
     * right-associative operators.
     */
    List<Map.Entry<Boolean, List<String>>> PRECEDENCE = List.of(
            Map.entry(true, List.of("^")),
            Map.entry(false, List.of("*", "/", "mod", "AND")),
            Map.entry(false, List.of("OR", "+", "-", ",")),
            Map.entry(false, List.of("=", "<>", ">=", "<=", ">", "<")),
            Map.entry(true, List.of(":="))
    );
}
