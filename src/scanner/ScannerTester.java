package scanner;

import ast.Environment;
import ast.Statement;
import parser.Parser;

import java.io.*;

/**
 * ScannerTester.java
 * @author Grant Yang
 * @version 2024.01.26
 * Tester class that uses the Scanner class to scan the two input test files
 * and prints out the resulting stream of tokens.
 */
public class ScannerTester
{
    /**
     * Entry point into the tester
     * @param args Command-line arguments
     * @throws FileNotFoundException If either of the two test cases cannot be read
     */
    public static void main(String[] args) throws IOException
    {
//        Scanner scan = new Scanner(new FileInputStream("scanner/scannerTestAdvanced.txt"));
//        while (scan.hasNext())
//            System.out.println(scan.nextToken());

        System.out.println("=======================================");
//        Scanner scan2 = new Scanner(new FileInputStream("scanner/ScannerTest.txt"));
        Scanner scan2 = new Scanner(new FileInputStream("ast/parserTest6.txt"));
        Environment env = new Environment();
        Parser parse = new Parser(scan2);
        Statement s = parse.parseStatement();
        System.out.println("exec1");
        s.exec(env);
        System.out.println("e2");
        s.exec(env);
//        while (scan2.hasNext())
//            System.out.println(scan2.nextToken());
//        System.out.println("end");
    }
}