package parser;

public class BoxedValue
{
    private static int counter = 0;
    private Object value;
    private String name = "var";

    public static BoxedValue newNamed(String name)
    {
        BoxedValue ret = new BoxedValue(null);
        ret.name = name + "$$" + counter++;
        return ret;
    }

    public BoxedValue(Object val)
    {
        if (val instanceof BoxedValue)
            throw new RuntimeException("cannot nest boxed values: " + val);
        value = val;
        name += "$" + counter++;
    }

    public Object get()
    {
        if (value instanceof BoxedValue)
            throw new RuntimeException("nested boxed value (on get): " + toString());
        return value;
    }

    public BoxedValue set(Object v)
    {
        if (value instanceof BoxedValue)
            throw new RuntimeException("nested boxed value (on set): " + toString());
        value = v;
        return this;
    }

    public int asInt()
    {
        return (Integer) get();
    }

    public boolean asBool()
    {
        return (Boolean) get();
    }

    public String toString()
    {
        return name + "{" + value + "}";
    }

    public boolean equals(Object rhs)
    {
        if (rhs instanceof BoxedValue)
            return get().equals(((BoxedValue) rhs).get());
        throw new RuntimeException("comparison between boxed and non-boxed value");
    }

    public static BoxedValue box(Object val)
    {
        return new BoxedValue(val);
    }
}
