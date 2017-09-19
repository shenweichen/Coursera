package patmat

import common._

/**
  * Assignment 4: Huffman coding
  *
  */
object Huffman extends App{

  /**
    * A huffman code is represented by a binary tree.
    *
    * Every `Leaf` node of the tree represents one character of the alphabet that the tree can encode.
    * The weight of a `Leaf` is the frequency of appearance of the character.
    *
    * The branches of the huffman tree, the `Fork` nodes, represent a set containing all the characters
    * present in the leaves below it. The weight of a `Fork` node is the sum of the weights of these
    * leaves.
    */
  abstract class CodeTree
  case class Fork(left: CodeTree, right: CodeTree, chars: List[Char], weight: Int) extends CodeTree
  case class Leaf(char: Char, weight: Int) extends CodeTree


  // Part 1: Basics
  def weight(tree: CodeTree): Int = tree match{
    case Fork(l, r, cs, w) => w
    case Leaf(c, w) => w
  }

  def chars(tree: CodeTree): List[Char] = tree match{
    case Fork(l, r, cs, w) => cs
    case Leaf(c, w) => List(c)
  }

  def makeCodeTree(left: CodeTree, right: CodeTree) =
    Fork(left, right, chars(left) ::: chars(right), weight(left) + weight(right))



  // Part 2: Generating Huffman trees

  /**
    * In this assignment, we are working with lists of characters. This function allows
    * you to easily create a character list from a given string.
    */
  def string2Chars(str: String): List[Char] = str.toList

  /**
    * This function computes for each unique character in the list `chars` the number of
    * times it occurs. For example, the invocation
    *
    *   times(List('a', 'b', 'a'))
    *
    * should return the following (the order of the resulting list is not important):
    *
    *   List(('a', 2), ('b', 1))
    *
    * The type `List[(Char, Int)]` denotes a list of pairs, where each pair consists of a
    * character and an integer. Pairs can be constructed easily using parentheses:
    *
    *   val pair: (Char, Int) = ('c', 1)
    *
    * In order to access the two elements of a pair, you can use the accessors `_1` and `_2`:
    *
    *   val theChar = pair._1
    *   val theInt  = pair._2
    *
    * Another way to deconstruct a pair is using pattern matching:
    *
    *   pair match {
    *     case (theChar, theInt) =>
    *       println("character is: "+ theChar)
    *       println("integer is  : "+ theInt)
    *   }
    */
  def times(chars: List[Char]): List[(Char, Int)] = {
    val pairEval: (Char, Int) = (chars.head, chars.count(_ == chars.head))
    if ((chars.filter(_ != chars.head)).isEmpty) List(pairEval)
    else List(pairEval) ::: times(chars.filter(_ != chars.head))
  }
  /**
    * Returns a list of `Leaf` nodes for a given frequency table `freqs`.
    *
    * The returned list should be ordered by ascending weights (i.e. the
    * head of the list should have the smallest weight), where the weight
    * of a leaf is the frequency of the character.
    */
  def makeOrderedLeafList(freqs: List[(Char, Int)]): List[Leaf] = {
    def insert(x: (Char, Int), xs: List[(Char, Int)]): List[(Char, Int)] = xs match{
      case List() => List(x)
      case y::ys => if(x._2 < y._2) x::xs else y::insert(x,ys)
    }
    def sort(xs: List[(Char, Int)]): List[(Char, Int)] = xs match{
      case List() => List()
      case y::ys => insert(y, sort(ys))
    }
    sort(freqs).map(x => Leaf(x._1, x._2))
  }

  /**
    * Checks whether the list `trees` contains only one single code tree.
    */
  def singleton(trees: List[CodeTree]): Boolean = trees.length == 1

  /**
    * The parameter `trees` of this function is a list of code trees ordered
    * by ascending weights.
    *
    * This function takes the first two elements of the list `trees` and combines
    * them into a single `Fork` node. This node is then added back into the
    * remaining elements of `trees` at a position such that the ordering by weights
    * is preserved.
    *
    * If `trees` is a list of less than two elements, that list should be returned
    * unchanged.
    */
  def combine(trees: List[CodeTree]): List[CodeTree] = trees match{
    case List() => List()
    case List(t) => List(t)
    case t1::t2::ts => (Fork(t1,t2,chars(t1):::chars(t2), weight(t1)+weight(t2)))::ts
  }

  /**
    * This function will be called in the following way:
    *
    *   until(singleton, combine)(trees)
    *
    * where `trees` is of type `List[CodeTree]`, `singleton` and `combine` refer to
    * the two functions defined above.
    *
    * In such an invocation, `until` should call the two functions until the list of
    * code trees contains only one single tree, and then return that singleton list.
    *
    * Hint: before writing the implementation,
    *  - start by defining the parameter types such that the above example invocation
    *    is valid. The parameter types of `until` should match the argument types of
    *    the example invocation. Also define the return type of the `until` function.
    *  - try to find sensible parameter names for `xxx`, `yyy` and `zzz`.
    */
  def until(xxx: List[CodeTree] => Boolean, yyy: List[CodeTree] => List[CodeTree])(zzz: List[CodeTree]): List[CodeTree] =
    if(xxx(zzz)) zzz else until(xxx, yyy)(yyy(zzz))

  /**
    * This function creates a code tree which is optimal to encode the text `chars`.
    *
    * The parameter `chars` is an arbitrary text. This function extracts the character
    * frequencies from that text and creates a code tree based on them.
    */
  def createCodeTree(chars: List[Char]): CodeTree = until(singleton, combine)(makeOrderedLeafList(times(chars))).head


  // Part 3: Decoding

  type Bit = Int

  /**
    * This function decodes the bit sequence `bits` using the code tree `tree` and returns
    * the resulting list of characters.
    */
  def decode(tree: CodeTree, bits: List[Bit]): List[Char] = {
    def loop(t: CodeTree, b: List[Bit]): (List[Char],List[Bit]) = t match {
      case Leaf(c, w) => (List(c),b)
      case Fork(l, r, cs, w) => if (b.head == 0) loop(l, b.tail) else loop(r, b.tail)
    }
    if(loop(tree, bits)._2.isEmpty) loop(tree, bits)._1 else loop(tree, bits)._1:::decode(tree, loop(tree, bits)._2)
  }

  /**
    * A Huffman coding tree for the French language.
    * Generated from the data given at
    *   http://fr.wikipedia.org/wiki/Fr%C3%A9quence_d%27apparition_des_lettres_en_fran%C3%A7ais
    */
  val frenchCode: CodeTree = Fork(Fork(Fork(Leaf('s',121895),Fork(Leaf('d',56269),Fork(Fork(Fork(Leaf('x',5928),Leaf('j',8351),List('x','j'),14279),Leaf('f',16351),List('x','j','f'),30630),Fork(Fork(Fork(Fork(Leaf('z',2093),Fork(Leaf('k',745),Leaf('w',1747),List('k','w'),2492),List('z','k','w'),4585),Leaf('y',4725),List('z','k','w','y'),9310),Leaf('h',11298),List('z','k','w','y','h'),20608),Leaf('q',20889),List('z','k','w','y','h','q'),41497),List('x','j','f','z','k','w','y','h','q'),72127),List('d','x','j','f','z','k','w','y','h','q'),128396),List('s','d','x','j','f','z','k','w','y','h','q'),250291),Fork(Fork(Leaf('o',82762),Leaf('l',83668),List('o','l'),166430),Fork(Fork(Leaf('m',45521),Leaf('p',46335),List('m','p'),91856),Leaf('u',96785),List('m','p','u'),188641),List('o','l','m','p','u'),355071),List('s','d','x','j','f','z','k','w','y','h','q','o','l','m','p','u'),605362),Fork(Fork(Fork(Leaf('r',100500),Fork(Leaf('c',50003),Fork(Leaf('v',24975),Fork(Leaf('g',13288),Leaf('b',13822),List('g','b'),27110),List('v','g','b'),52085),List('c','v','g','b'),102088),List('r','c','v','g','b'),202588),Fork(Leaf('n',108812),Leaf('t',111103),List('n','t'),219915),List('r','c','v','g','b','n','t'),422503),Fork(Leaf('e',225947),Fork(Leaf('i',115465),Leaf('a',117110),List('i','a'),232575),List('e','i','a'),458522),List('r','c','v','g','b','n','t','e','i','a'),881025),List('s','d','x','j','f','z','k','w','y','h','q','o','l','m','p','u','r','c','v','g','b','n','t','e','i','a'),1486387)

  /**
    * What does the secret message say? Can you decode it?
    * For the decoding use the'frenchCode' Huffman tree defined above.
    */
  val secret: List[Bit] = List(0,0,1,1,1,0,1,0,1,1,1,0,0,1,1,0,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,1)

  /**
    * Write a function that returns the decoded secret
    */
  def decodedSecret: List[Char] = decode(frenchCode, secret)
  println(decodedSecret)

  // Part 4a: Encoding using Huffman tree

  /**
    * This function encodes `text` using the code tree `tree`
    * into a sequence of bits.
    */
  def encode(tree: CodeTree)(text: List[Char]): List[Bit] = {
    def isSubsetOfMyTree(h: Char, t: CodeTree): Boolean = t match{
      case Leaf(c,w) => c == h
      case Fork(l,r,cs,w) => isSubsetOfMyTree(h, l) || isSubsetOfMyTree(h, r)
    }
    def encodeOneChar(t: CodeTree)(oneC: Char): List[Bit] = t match{
      case Leaf(_,_) => List()
      case Fork(l,_,_,_) if(isSubsetOfMyTree(oneC, l))=>List(0):::encodeOneChar(l)(oneC)
      case Fork(_,r,_,_) => List(1):::encodeOneChar(r)(oneC)
    }
    text match {
      case List() => List()
      case x::xs => encodeOneChar(tree)(x):::encode(tree)(xs)
    }
  }

  val myCode =
    Fork(
      Leaf('a',8),
      Fork(
        Fork(
          Leaf('b',3),
          Fork(
            Leaf('c',1),
            Leaf('d',1),
            "cd".toList,
            2),
          "bcd".toList,5),
        Fork(
          Fork(
            Leaf('e',1),
            Leaf('f',1),
            "ef".toList,
            2),
          Fork(
            Leaf('g',1),
            Leaf('h',1),
            "gh".toList,
            2),
          "efgh".toList,
          4),
        "bcdefgh".toList,
        9),
      "abcdefgh".toList,
      17
    )
  println(encode(myCode)("ahbad".toList))
  // Part 4b: Encoding using code table

  type CodeTable = List[(Char, List[Bit])]

  /**
    * This function returns the bit sequence that represents the character `char` in
    * the code table `table`.
    */
  def codeBits(table: CodeTable)(char: Char): List[Bit] = if(table.head._1 == char) table.head._2 else codeBits(table.tail)(char)

  /**
    * Given a code tree, create a code table which contains, for every character in the
    * code tree, the sequence of bits representing that character.
    *
    * Hint: think of a recursive solution: every sub-tree of the code tree `tree` is itself
    * a valid code tree that can be represented as a code table. Using the code tables of the
    * sub-trees, think of how to build the code table for the entire tree.
    */
  def convert(tree: CodeTree): CodeTable = {
    def conv(t: CodeTree)(bits: List[Bit]): CodeTable = t match{
      case Leaf(c,w) => List((c, bits ::: List()))
      case Fork(l,r,cs,w) => conv(l)(bits ::: List(0)) ::: conv(r)(bits ::: List(1))
    }
    conv(tree)(List())
  }

  /**
    * This function takes two code tables and merges them into one. Depending on how you
    * use it in the `convert` method above, this merge method might also do some transformations
    * on the two parameter code tables.
    */
  def mergeCodeTables(a: CodeTable, b: CodeTable): CodeTable = a union b

  /**
    * This function encodes `text` according to the code tree `tree`.
    *
    * To speed up the encoding process, it first converts the code tree to a code table
    * and then uses it to perform the actual encoding.
    */
  def quickEncode(tree: CodeTree)(text: List[Char]): List[Bit] = {
    val myTree = convert(tree)
    def search(t: CodeTable)(c: Char):List[Bit] = {
      if(t.head._1 == c) t.head._2 else search(t.tail)(c)
    }
    text match{
      case List() => List()
      case x::xs => search(myTree)(x):::quickEncode(tree)(text.tail)
    }
  }
}
