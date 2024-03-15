package ast;

import parser.BoxedValue;

import java.util.Map;

import static parser.BoxedValue.box;

public interface OperatorSAM
{
    BoxedValue apply(Environment env, Expression left, Expression right);

    OperatorSAM ADD = (e, a, b) -> box(a.eval(e).asInt() + b.eval(e).asInt());
    OperatorSAM POW = (e, a, b) -> box((int) Math.pow(a.eval(e).asInt(), b.eval(e).asInt()));
    OperatorSAM MUL = (e, a, b) -> box(a.eval(e).asInt() * b.eval(e).asInt());
    OperatorSAM DIV = (e, a, b) -> box(a.eval(e).asInt() / b.eval(e).asInt());
    OperatorSAM MOD = (e, a, b) -> box(a.eval(e).asInt() % b.eval(e).asInt());
    OperatorSAM AND = (e, a, b) -> box(a.eval(e).asBool() && b.eval(e).asBool());
    OperatorSAM OR = (e, a, b) -> box(a.eval(e).asBool() || b.eval(e).asBool());
    OperatorSAM SUB = (e, a, b) -> box(a.eval(e).asInt() - b.eval(e).asInt());
    OperatorSAM CONCAT = (e, a, b) -> box(a.eval(e).get().toString() + b.eval(e).get().toString());
    OperatorSAM EQ = (e, a, b) -> box(a.eval(e).get().equals(b.eval(e).get()));
    OperatorSAM NE = (e, a, b) -> box(!a.eval(e).get().equals(b.eval(e).get()));
    OperatorSAM GE = (e, a, b) -> box(a.eval(e).asInt() >= b.eval(e).asInt());
    OperatorSAM LE = (e, a, b) -> box(a.eval(e).asInt() <= b.eval(e).asInt());
    OperatorSAM LT = (e, a, b) -> box(a.eval(e).asInt() < b.eval(e).asInt());
    OperatorSAM GT = (e, a, b) -> box(a.eval(e).asInt() > b.eval(e).asInt());
    OperatorSAM SET = (e, a, b) ->
    {
        Object v = b.eval(e).get();
        return a.eval(e).set(v);
    };
}
