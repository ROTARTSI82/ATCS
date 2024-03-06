package scanner;

/**
 * Token.java
 * @author Grant Yang
 * @version 2024.01.25
 * Token record that represents lexemes emitted by the Scanner,
 * a record that contains information including source information and token type.
 *
 * @param line Line number of the input source this token was found in
 * @param col Column number of the input source this token was found in
 * @param type The type of token. See Token.Type documentation.
 * @param content String content of the token.
 */
public record Token(int line, int col, Type type, String content)
{

    /**
     * The type of the token, telling the parser how it should interpret it.
     * Places implicit restrictions on the Token's `content` value, determined during scanning.
     */
    public enum Type
    {
        /**
         * Numeric integer literal. Regex is `[0-9]+`
         */
        Numeric,

        /**
         * Alpha numeric identifier. Regex is `[A-z][A-z0-9]*`
         */
        Identifier,

        /**
         * Operator of special characters.
         * See full operator table definitions in `Scanner::initializeTables`
         */
        Operator,

        /**
         * End-Of-File token. Emitted when the raw input stream reaches
         * its end, or when a period '.' character is encountered.
         */
        EOF,

        /**
         * Keyword, a special sub-case of the Identifier type.
         * See full keyword table definition in `Scanner::initializeTables`
         */
        Keyword,

        /**
         * content is a user-specified string literal, literally.
         */
        StringLiteral
    }
}
