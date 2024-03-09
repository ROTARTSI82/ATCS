package parser;

/**
 * PascalArray.java implements an array that supports special
 * indexing, as required by Pascal. The start index can be set
 * to any arbitrary integer.
 *
 * @author Grant Yang
 * @version 2024.03.06
 */
public class PascalArray
{
    private static int counter = 0;
    private final BoxedValue[] data;
    private final int startIndex;
    private final int tag;

    /**
     * Construct and allocate a new array with indices ranging from lo to hi, inclusive.
     * It is initially filled with null Objects.
     * @param lo The lowest valid integer index
     * @param hi The highest valid integer index
     */
    public PascalArray(int lo, int hi)
    {
        startIndex = lo;
        data = new BoxedValue[hi - lo + 1];
        for (int i = lo; i <= hi; i++)
            data[i - startIndex] = BoxedValue.newNamed("arr_" + counter + "$" + i);
        tag = counter++;
    }

    /**
     * Retrieves the element at index idx (indexing according to Pascal's convention)
     * @param idx Index to read
     * @return A reference to the Object at idx
     */
    public BoxedValue at(int idx)
    {
        return data[idx - startIndex];
    }

    public String toString()
    {
        return "arr_" + tag;
    }
}
