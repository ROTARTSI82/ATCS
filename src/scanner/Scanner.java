package scanner;

import java.io.*;
import java.util.*;

/**
 * Scanner is a simple scanner for Compilers and Interpreters (2014-2015) lab exercise 1
 * @author Grant Yang
 * @version 2024.01.22
 * Scanner tokenizes the raw input stream of characters into a stream
 * of tokens, generating information about the type and source location of the tokens
 * and discarding whitespace and comment blocks.
 * Usage:
 * Scanner scan = new Scanner(file);
 * while (scan.hasNext())
 *     System.out.println(scan.nextToken());
 */
public class Scanner
{
    private BufferedReader in;
    private char currentChar;
    private boolean eof;

    private int colNo = 1, lineNo = 1;
    private static Set<String> operators, partialOperators, keywords;
    private static boolean initialized = false;

    /**
     * Initializes the internal tables used to determine operators and keywords.
     * This method is called automatically whenever tables need to be used, and
     * it does nothing if tables are already initialized.
     * Precondition: none
     * Postcondition: Tables are populated
     */
    private static void initializeTables()
    {
        if (initialized)
            return;

        operators = Set.of(
                "+", "-", "*", "/*", "*/", "/", "//", "(", ")",
                ":", ":=", "<=", ">=", "<", ">", ".", "=", "%", ";", "<>"
        );
        partialOperators = Set.of(":");
        keywords = Set.of("BEGIN", "END", "VAR", "WHILE", "RETURN", "PROCEDURE", "DO", "WRITELN");
        initialized = true;
    }

    /**
     * Scanner constructor for construction of a scanner that 
     * uses an InputStream object for input.
     * Usage: 
     * FileInputStream inStream = new FileInputStream(new File(<file name>);
     * Scanner lex = new Scanner(inStream);
     * @param inStream the input stream to use
     * Precondition: inStream is not null and in a valid state
     * Postcondition: Scanner is initialized
     */
    public Scanner(InputStream inStream)
    {
        in = new BufferedReader(new InputStreamReader(inStream));
        eof = false;
        getNextChar();
        initializeTables();
    }
    
    /**
     * Scanner constructor for constructing a scanner that 
     * scans a given input string.  It sets the end-of-file flag an then reads
     * the first character of the input string into the instance field currentChar.
     * Usage: Scanner lex = new Scanner(input_string);
     * @param inString the string to scan
     * Precondition: none
     * Postcondition: Scanner is initialized
     */
    public Scanner(String inString)
    {
        in = new BufferedReader(new StringReader(inString));
        eof = false;
        getNextChar();
        initializeTables();
    }

    /**
     * A pure function to determine if a character is a numeric digit,
     * according to the regex `[0-9]`
     * @param d character to check
     * @return True if d is a digit, false otherwise.
     */
    public static boolean isDigit(char d)
    {
        return '0' <= d && d <= '9';
    }

    /**
     * A pure function to determine if a character is a letter,
     * according to the regex `[a-zA-Z]`
     * @param d character to check
     * @return True if d is a letter, false otherwise.
     */
    public static boolean isLetter(char d)
    {
        return 'a' <= d && d <= 'z' || 'A' <= d && d <= 'Z';
    }

    /**
     * A pure function to determine if a character is whitespace,
     * according to the regex `[\n\r\t ]`
     * @param d character to check
     * @return True if d is a whitespace character, false otherwise
     */
    public static boolean isWhitespace(char d)
    {
        return "\n\r\t ".indexOf(d) != -1;
    }

    /**
     * Gets the next character from the raw underlying input stream.
     * Precondition: The underlying input stream is in a valid state and is not at EOF.
     * Postcondition: The input stream has advanced by one character, and
     *                `currentChar` is updated to reflect the new cursor position.
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
                throw new IOException("line %d:%d unicode characters ('%d') is not supported"
                        .formatted(lineNo, colNo, value));

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
     * Consume a single character from the input stream,
     * asserting that the character read matches the expected one.
     * Precondition: The next character is `expected`
     * Postcondition: The stream is advanced by one character, and
     *                `currentChar` is updated to reflect the new cursor position.
     * @param expected The character to expect
     * @throws ScanErrorException if the next character does not match expected
     */
    private void eat(char expected) throws ScanErrorException
    {
        if (eof)
            compileThrow("premature eof: expected '%c'".formatted(expected));
        if (currentChar != expected)
            compileThrow("expected '%c' and got '%c'".formatted(expected, currentChar));
        getNextChar();
    }

    /**
     * Convenience method to throw an error message, including information about
     * the source location that the error was encountered at.
     * @param message error that was encountered
     * @throws ScanErrorException when called
     */
    private void compileThrow(String message) throws ScanErrorException
    {
        throw new ScanErrorException("line %d:%d %s".formatted(lineNo, colNo, message));
    }

    /**
     * Convenience method for constructing a new token,
     * automatically filling out the source location the token was found at.
     *
     * @param type Token's type, see documentation for Token.Type enum
     * @param content String content of the token
     * @return Completed token object
     */
    private Token newToken(Token.Type type, String content)
    {
        return new Token(lineNo, colNo, type, content);
    }

    /**
     * Pure function that checks if this scanner has reached the end of file.
     * This function may return true even if the next token is EOF, as the scanner
     * must still process the remaining to determine that the next token is EOF.
     * @return True if there may be more tokens to read, false if there are guaranteed
     *         to be no more tokens left.
     */
    public boolean hasNext()
    {
        return !eof;
    }

    /**
     * Parses and emits the next token in the raw input stream, consuming it in the process.
     * Precondition: None
     * Postcondition: The internal input stream has advanced by any number of tokens,
     *                possibly consuming the entire stream.
     * @return The next token in the stream, or a Token.Type.EOF if there are no more
     *         tokens left in the stream.
     */
    public Token nextToken()
    {
        try
        {
            while (isWhitespace(currentChar) && !eof)
                eat(currentChar);

            eof = eof || currentChar == '.';
            if (eof)
                return newToken(Token.Type.EOF, "EOF");

            if (isLetter(currentChar))
                return scanIdentifier();
            else if (isDigit(currentChar))
                return scanNumber();

            return skipComments(scanOperator());
        }
        catch (ScanErrorException error)
        {
            error.printStackTrace();
            if (!eof)
            {
                try
                {
                    eat(currentChar);
                }
                catch (ScanErrorException e)
                {
                    // ignore
                }
            }

            return nextToken();
        }
    }

    /**
     * Private helper method called on all Operator Tokens encountered
     * to transparently skip over any comments. Comment-related tokens
     * should only ever exist in this function, and they should never leave this code.
     * Precondition: Stream is not at EOF and contains valid comment syntax
     * Postcondition: Stream is advanced to the first character after the comment if
     *                `op` is a comment operator, otherwise nothing changes.
     * @param op The operator token to handle
     * @return The original operator token, or the token after it if
     *         op was a comment operator.
     * @throws ScanErrorException if end-of-file is reached within a comment
     */
    private Token skipComments(Token op) throws ScanErrorException
    {
        if (op.content().equals("//"))
        {
            while (currentChar != '\n')
                eat(currentChar);
            return nextToken();
        }
        if (op.content().equals("/*"))
        {
            scanBlockComment();
            return nextToken();
        }
        if (op.content().equals("*/"))
            compileThrow("unbalanced closing comment '*/'");

        return op;
    }

    /**
     * Advances the input stream until a valid closing comment operator (star slash)
     * is met, discarding the content until then. This function will correctly handle
     * nested block comments, which is unusual behavior but cool I guess.
     * Precondition: The stream is positioned inside a block comment
     * Postcondition: The stream is advanced to the first character after the end of the comment.
     * @throws ScanErrorException if EOF is reached
     */
    private void scanBlockComment() throws ScanErrorException
    {
        int pending = 1;
        while (pending > 0)
        {
            while (currentChar != '/' && currentChar != '*')
                eat(currentChar);
            Token op2 = scanOperator();
            if (op2.content().equals("*/"))
                pending--;
            else if (op2.content().equals("/*"))
                pending++;
        }
    }

    /**
     * Consumes a number from the input stream, returning it in a Token.
     * Numbers are defined by the regex `[0-9]+`
     * Precondition: The stream is positioned at the beginning of a valid number
     * Postcondition: The stream is advanced past the number
     * @return A token with `Token::Type::Numeric`
     * @throws ScanErrorException if the precondition is not met or EOF is reached
     */
    private Token scanNumber() throws ScanErrorException
    {
        if (!isDigit(currentChar))
            compileThrow("number must start with digit, not '%c'".formatted(currentChar));

        StringBuilder ret = new StringBuilder();
        while (isDigit(currentChar))
        {
            ret.append(currentChar);
            eat(currentChar);
        }

        return newToken(Token.Type.Numeric,  ret.toString());
    }

    /**
     * Consumes an identifier or keyword from the input stream, returning it in a Token.
     * Identifiers are defined by the regex `[a-zA-Z][A-Za-z0-9]*`
     * See `Scanner::initializeTables` for a full list of identifiers that are considered keywords
     * Precondition: The stream is positioned at the beginning of a valid identifier
     * Postcondition: The stream is advanced past the identifier
     * @return A token with `Token::Type::Identifier` or `Token::Type::Keyword`
     * @throws ScanErrorException if the precondition is not met or EOF is reached
     */
    private Token scanIdentifier() throws ScanErrorException
    {
        if (!isLetter(currentChar))
            compileThrow("identifier must start with letter, not '%c'".formatted(currentChar));

        StringBuilder ret = new StringBuilder();
        while (isLetter(currentChar) || isDigit(currentChar))
        {
            ret.append(currentChar);
            eat(currentChar);
        }

        Token.Type type = keywords.contains(ret.toString()) ?
                Token.Type.Keyword : Token.Type.Identifier;
        return newToken(type, ret.toString());
    }

    /**
     * Consumes an operator from the input stream, returning it in a Token.
     * See `Scanner::initializeTables` for a full list of operators that are recognized.
     * Precondition: The stream is positioned at the beginning of a valid operator
     * Postcondition: The stream is advanced past the operator
     * @return A token with `Token::Type::Operator`
     * @throws ScanErrorException if the precondition is not met or EOF is reached
     */
    private Token scanOperator() throws ScanErrorException
    {
        String ret = "" + currentChar;
        while (operators.contains(ret))
        {
            eat(currentChar);
            ret += currentChar;
        }

        if (ret.length() == 1)
            compileThrow("'%c' is not a valid operator".formatted(currentChar));

        ret = ret.substring(0, ret.length() - 1);
        if (partialOperators.contains(ret))
            compileThrow("'%s' is not a valid operator (partial: insert a space to disambiguate)"
                    .formatted(ret));

        return newToken(Token.Type.Operator, ret);
    }
}
