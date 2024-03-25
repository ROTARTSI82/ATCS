package parser;

import javax.swing.*;

/**
 * BoxedValue.java
 * @author Grant Yang
 * @version 2024.03.12
 * This class represents a Pascal value that appears as the
 * code is being interpreted. This abstraction is necessary
 * to unify the treatment of r-values and l-values, and this
 * effectively acts as a pointer/reference to a value with additional
 * attached information that can be passed around and used to modify
 * the referenced object.
 * BoxedValues can also never be nested.
 */
public class BoxedValue
{
    private static int counter = 0;
    private Object value;
    private String name;

    /**
     * Construct a new BoxedValue from a name.
     * This is useful for debugging values that are variables,
     * as they may also be used as l-values as well as r-values.
     * @param name The name to give the box. This boxed value's name will be
     *             name$##, where ## is a unique number given to this box.
     * @return A new Box with null value.
     */
    public static BoxedValue newNamed(String name)
    {
        BoxedValue ret = new BoxedValue();
        ret.value = null;
        ret.name = name + "$" + counter++;
        return ret;
    }

    /**
     * Get the internal value of this box.
     * @return The internal value
     */
    public Object get()
    {
        if (value instanceof BoxedValue)
            throw new RuntimeException("nested boxed value (on get): " + toString());
        return value;
    }

    /**
     * Set the internal value of this box.
     * @param v The new value to replace it with
     * @return A reference to this BoxedValue
     */
    public BoxedValue set(Object v)
    {
        if (value instanceof BoxedValue)
            throw new RuntimeException("nested boxed value (on set): " + toString());
        value = v;
        return this;
    }

    /**
     * Convenience methods to get the internal value
     * and cast it to the desired primitive type.
     * @return The internal value of this box
     */

    public int asInt()
    {
        return (Integer) get();
    }

    public boolean asBool()
    {
        return (Boolean) get();
    }

    /**
     * Convert this boxed value to a string,
     * printing out both the name and the value it contains.
     * @return A string of the form "name{value}"
     */
    public String toString()
    {
        return name + "{" + value + "}";
    }

    /**
     * Convenience method to construct a new boxed value with
     * the specified internal value.
     * @param val Value for the box to contain
     * @return A new box containing the specified value
     */
    public static BoxedValue box(Object val)
    {
        if (val instanceof BoxedValue)
            throw new RuntimeException("cannot nest boxed values: " + val);
        BoxedValue ret = new BoxedValue();
        ret.value = val;
        ret.name = "$" + counter++;
        return ret;
    }
}
