package ast;

import parser.BoxedValue;

import java.util.List;
import java.util.Map;

import static parser.BoxedValue.box;

public interface OperatorSAM
{
    BoxedValue apply(Environment env, Expression left, Expression right);

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

    List<Map.Entry<Boolean, List<String>>> PRECEDENCE = List.of(
            Map.entry(true, List.of("^")),
            Map.entry(false, List.of("*", "/", "mod", "AND")),
            Map.entry(false, List.of("OR", "+", "-", ",")),
            Map.entry(false, List.of("=", "<>", ">=", "<=", ">", "<")),
            Map.entry(true, List.of(":="))
    );
}
