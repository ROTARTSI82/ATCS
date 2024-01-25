package scanner;

import java.io.*;
import java.util.*;

/**
 * Scanner is a simple scanner for Compilers and Interpreters (2014-2015) lab exercise 1
 * @author Grant Yang
 * @version 2024.01.22
 *  
 * Usage:
 * <Insert a comment that shows how to use this object>
 *
 */
public class Scanner
{
    private BufferedReader in;
    private char currentChar;
    private boolean eof;

    private int colNo = 1, lineNo = 1;

    /**
     * Scanner constructor for construction of a scanner that 
     * uses an InputStream object for input.  
     * Usage: 
     * FileInputStream inStream = new FileInputStream(new File(<file name>);
     * Scanner lex = new Scanner(inStream);
     * @param inStream the input stream to use
     */
    public Scanner(InputStream inStream)
    {
        in = new BufferedReader(new InputStreamReader(inStream));
        eof = false;
        getNextChar();
    }
    /**
     * Scanner constructor for constructing a scanner that 
     * scans a given input string.  It sets the end-of-file flag an then reads
     * the first character of the input string into the instance field currentChar.
     * Usage: Scanner lex = new Scanner(input_string);
     * @param inString the string to scan
     */
    public Scanner(String inString)
    {
        in = new BufferedReader(new StringReader(inString));
        eof = false;
        getNextChar();
    }

    public static boolean isDigit(char d)
    {
        return '0' <= d && d <= '9';
    }

    public static boolean isLetter(char d)
    {
        return 'a' <= d && d <= 'z' || 'A' <= d && d <= 'Z';
    }

    public static boolean isWhitespace(char d)
    {
        return "\n\r\t ;".indexOf(d) != -1;
    }

    /**
     * Method: getNextChar
     */
    private void getNextChar()
    {
        try
        {
            int value = in.read();
            if (value == -1)
                eof = true;
            else if (value < 256)
                currentChar = (char) value;
            else
                throw new IOException("unicode is not supported");

            if (currentChar == '\n')
            {
                lineNo++;
                colNo = 0;
            }
            else
                colNo++;
        }
        catch (IOException e)
        {
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Method: eat
     * @param expected
     */
    private void eat(char expected) throws ScanErrorException
    {
        if (eof)
            throw new ScanErrorException("premature eof: expected %c".formatted(expected));
        if (currentChar != expected)
            throw new ScanErrorException("line %d:%d expected '%c' and got '%c'"
                    .formatted(lineNo, colNo, expected, currentChar));
        getNextChar();
    }

    private void compileThrow(String message) throws ScanErrorException
    {
        throw new ScanErrorException(("line %d:%d " + message).formatted(lineNo, colNo, currentChar));
    }

    /**
     * Method: hasNext
     * @return
     */
    public boolean hasNext()
    {
        return !eof;
    }

    /**
     * Method: nextToken
     * @return
     */
    public String nextToken() throws ScanErrorException
    {
        while (isWhitespace(currentChar) && !eof)
            eat(currentChar);

        if (eof)
            return "END-EOF";

        if (isLetter(currentChar))
            return scanIdentifier();
        else if (isDigit(currentChar))
            return scanNumber();

        String op = scanOperator();
        // todo: rewrite this into scanOperator properly
        if (op.equals("//"))
        {
            while (currentChar != '\n')
                eat(currentChar);
            return nextToken();
        }

        if (op.equals("/*"))
        {
            int pending = 1;
            while (pending > 0)
            {
                while (currentChar != '/' && currentChar != '*')
                    eat(currentChar);
                String op2 = scanOperator();
                if (op2.equals("*/"))
                    pending--;
                else if (op2.equals("/*"))
                    pending++;
            }

            return nextToken();
        }
        if (op.equals("*/"))
            throw new ScanErrorException("line %d:%d unbalanced closing comment '*/'"
                    .formatted(lineNo, colNo));

        return op;
    }

    private String scanNumber() throws ScanErrorException
    {
        if (!isDigit(currentChar))
            compileThrow("number must start with digit, not '%c'");

        StringBuilder ret = new StringBuilder();
        do
        {
            ret.append(currentChar);
            eat(currentChar);
        } while (isDigit(currentChar));

        return ret.toString();
    }

    private String scanIdentifier() throws ScanErrorException
    {
        if (!isLetter(currentChar))
            compileThrow("identifier must start with letter, not '%c'");

        StringBuilder ret = new StringBuilder();
        do
        {
            ret.append(currentChar);
            eat(currentChar);
        } while (isLetter(currentChar) || isDigit(currentChar));

        return ret.toString();
    }

    private String scanOperator() throws ScanErrorException
    {
        Set<String> operators = Set.of(
                "+", "-", "*", "/*", "*/", "/", "//", "(", ")",
                ":", ":=", "<=", ">=", "<", ">", "."
        );

        String ret = "";
        while (true)
        {
            ret += currentChar;
            if (!operators.contains(ret))
                break;
            eat(currentChar);
        }

        if (ret.length() == 1)
            compileThrow("'%c' is not a valid operator");

        return ret.substring(0, ret.length() - 1);
    }
}
