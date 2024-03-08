package parser;

public class BoxedValue
{
    private final static boolean DEBUG = false;
    private Object value;
    private String name = "ANON";

    public static BoxedValue newNamed(String name)
    {
        BoxedValue ret = new BoxedValue(null);
        ret.name = name;
        return ret;
    }

    public BoxedValue(Object val)
    {
        value = val;
    }

    public Object get()
    {
        if (value instanceof BoxedValue)
            return ((BoxedValue) value).get();
        return value;
    }

    public Object set(Object v)
    {
        if (value instanceof BoxedValue)
            return ((BoxedValue) value).set(v);
        value = v;
        return v;
    }

    public int asInt()
    {
        if (value instanceof BoxedValue)
            return ((BoxedValue) value).asInt();
        return (Integer) value;
    }

    public boolean asBool()
    {
        if (value instanceof BoxedValue)
            return ((BoxedValue) value).asBool();
        return (Boolean) value;
    }

    public String toString()
    {
        if (DEBUG)
            return "box[" + value.toString() + "]";
        return value.toString();
    }

    public boolean equals(Object rhs)
    {
        if (rhs instanceof BoxedValue)
            return value.equals(((BoxedValue) rhs).value);
        return super.equals(rhs);
    }
}
