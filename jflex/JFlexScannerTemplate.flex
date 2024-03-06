package scanner;

/**
* This file defines a simple lexer for the compilers course 2017-2018. It
* defines regular expressions to scan over a list of harker students information.
* We have used emailChars to define the possible values that can be a character
* in an email address. 
* @author Grant Yang, Aditya Ramanathan
* @version 2/8/24
*/
import java.io.*;


%%
/* lexical functions */
/* The class will be called scanner and the function to get the next
 * token is called nextToken.  
 */
%class Scanner
%unicode
%line
%public
%function nextToken
/*  return String objects - the actual lexemes */
/*  returns the String "EOF" at end of file */
%type String
%eofval{
return "EOF";
%eofval}


/**
 * Pattern definitions. We define a macro for an email character, which is
 * used frequently in our regular expressions. The macro for a digit is already
 * defined by \d, and we use this frequently as well
 */
 
emailChars = [a-zA-Z0-9_\.]+

%%
/**
 * lexical rules/regular expressions for the scanner to use. 
 */

\"{emailChars}+\@{emailChars}(\.{emailChars}){1,2}\" { return "EMAIL " + yytext(); }
\"\+?1? ?(\d{3}|\(\d{3}\))[- ]?\d{3}[- ]?\d{4}\" { return "PHONE " + yytext(); }
\"(true|false)\" { return "BOOL " + yytext(); }
\"\d+\" { return "DIGIT " + yytext(); }
\{|\}|\[|\]|\:|\, { return "SEP " + yytext(); }
\"[^\"]*\" { return "QUOTE " + yytext(); }
. { return "UNREC " + yytext(); }
