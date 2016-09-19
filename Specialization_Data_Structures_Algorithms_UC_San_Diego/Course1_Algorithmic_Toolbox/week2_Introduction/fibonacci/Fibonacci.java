import java.util.Scanner;

public class Fibonacci {
  private static long calc_fib(int n) {
    if (n <= 1)
      return n;

    return calc_fib(n - 1) + calc_fib(n - 2);
  }
  private static long fast_fib(int n){
    if (n<=1)
      return n;
    int a = 0;
    int b = 0;
    int temp =0;
    for (int i =2 ;i<=n ;i++){
      temp = a+b;
      a =b ;
      b =temp;
    }
    return temp;
  }
  public static void main(String args[]) {
    Scanner in = new Scanner(System.in);
    int n = in.nextInt();

    System.out.println(calc_fib(n));
  }
}
