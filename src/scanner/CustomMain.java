package scanner;

import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class CustomMain
{
    public static void main(String[] args) throws FileNotFoundException, ScanErrorException
    {
        Scanner scan = new Scanner(new FileInputStream("scanner/scannerTestAdvanced.txt"));

        while (scan.hasNext())
            System.out.println(scan.nextToken());
    }
}