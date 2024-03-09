package parser;

public class BoxedValue
{
    private static int counter = 0;
    private Object value;
    private String name = "var";

    public static BoxedValue newNamed(String name)
    {
        BoxedValue ret = new BoxedValue(null);
        ret.name = name;
        return ret;
    }

    public BoxedValue(Object val)
    {
        value = val;
        name += "$" + counter++;
//        name += "{" + val + "}$" + counter++;
    }

    public Object get()
    {
        if (value instanceof BoxedValue)
            throw new RuntimeException("nexted boxed value (on get): " + toString());
//            return ((BoxedValue) value).get();
        return value;
    }

    public Object set(Object v)
    {
        if (value instanceof BoxedValue)
            throw new RuntimeException("nested boxed value (on set): " + toString());
//            return ((BoxedValue) value).set(v);
        value = v;
        return v;
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
}
