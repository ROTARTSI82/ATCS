package ast;

import parser.BoxedValue;

import java.util.HashMap;
import java.util.Map;

public class Environment
{
    private final Map<String, BoxedValue> variables = new HashMap<>();

    public void setVariable(String name, Object rawValue)
    {
        if (rawValue instanceof BoxedValue)
            throw new RuntimeException("setVariable() takes raw value, not boxed value");
        if (!variables.containsKey(name))
            variables.put(name, BoxedValue.box(rawValue));
        else
            variables.get(name).set(rawValue);
    }

    public BoxedValue getVariable(String name)
    {
        if (!variables.containsKey(name))
            variables.put(name, BoxedValue.newNamed(name));
        return variables.get(name);
    }
}
