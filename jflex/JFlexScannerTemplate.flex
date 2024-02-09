package scanner;

/**
* This file defines a simple lexer for the compilers course 2017-2018
* Comment this file
*/
import java.io.*;


%%
/* lexical functions */
/* specify that the class will be called Scanner and the function to get the next
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
 * Pattern definitions
 */
 
emailChars = [a-zA-Z0-9_]+

%%
/**
 * lexical rules
 */

\"[a-zA-Z0-9_\.]+\@{emailChars}(\.{emailChars}){1,2}\" { return "EMAIL " + yytext(); }
\"+?1? ?(\d{3}|(\(\d{3}\)))[- ]?\d{3}[- ]?\d{4}\" { return "PHONE " + yytext(); }
\"(true|false)\" { return "BOOL " + yytext(); }
\"\d+\" { return "DIGIT " + yytext(); }
\{|\}|\[|\]|\:|\, { return "SEP " + yytext(); }
\"[^\"]*\" { return "QUOTE " + yytext(); }

. { return "UNREC " + yytext(); }
