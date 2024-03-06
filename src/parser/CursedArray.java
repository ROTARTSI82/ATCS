package parser;

public class CursedArray
{
    private final Object[] data;
    private final int startIndex;

    public CursedArray(int lo, int hi)
    {
        startIndex = lo;
        data = new Object[hi - lo + 1];
    }

    public Object at(int idx)
    {
        return data[idx - startIndex];
    }

    public void set(int idx, Object obj)
    {
        data[idx - startIndex] = obj;
    }
}
