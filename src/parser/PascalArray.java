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
    private final Object[] data;
    private final int startIndex;

    /**
     * Construct and allocate a new array with indices ranging from lo to hi, inclusive.
     * It is initially filled with null Objects.
     * @param lo The lowest valid integer index
     * @param hi The highest valid integer index
     */
    public PascalArray(int lo, int hi)
    {
        startIndex = lo;
        data = new Object[hi - lo + 1];
    }

    /**
     * Retrieves the element at index idx (indexing according to Pascal's convention)
     * @param idx Index to read
     * @return A reference to the Object at idx
     */
    public Object at(int idx)
    {
        return data[idx - startIndex];
    }

    /**
     * Replaces the element at index idx (indexing according to Pascal's convention)
     * @param idx Index to replace
     * @param obj The new value to replace it with
     */
    public void set(int idx, Object obj)
    {
        data[idx - startIndex] = obj;
    }
}
