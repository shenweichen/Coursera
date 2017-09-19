package recfun

object Main {
  def main(args: Array[String]) {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(pascal(col, row) + " ")
      println()
    }
  }

  /**
   * Exercise 1
   */
    def pascal(c: Int, r: Int): Int =
      if ( c==0 || c==r)1
      else pascal(c-1,r-1)+pascal(c,r-1)

  
  /**
   * Exercise 2
   */

  def balance(chars: List[Char]): Boolean = {
    def judge (chars: List[Char], state: Int): Boolean  = {
      if (chars.isEmpty) if (state == 0)
        return true
      else
        return false
      if (state < 0)
      return false
      if (chars.head == '(')
      judge(chars.tail, state + 1)
      else if (chars.head == ')')
      judge(chars.tail, state - 1)
      else
      judge(chars.tail, state)
    }
    judge(chars,0)
  }

  
  /**
   * Exercise 3
   */
    def countChange(money: Int, coins: List[Int]): Int = {
      if (money == 0)
        return 1
      if (coins.isEmpty)
      return 0

      var sum = 0
      for(i<-0 to money/coins.head){
        sum = countChange(money-coins.head*i,coins.tail)+sum;
      }

     sum

    }



}
