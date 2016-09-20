# [Data Structures](https://www.coursera.org/learn/data-structures/home/welcome)

## week1 Basic Data Structures

* 顺序表,链表
* 栈和队列
* 树和树的遍历
* 动态数组
* 平摊分析 **这部分没搞懂，以后重新做下作业**

| Assignments                             | Tags       | C++  | Python                      | Java                           |
| --------------------------------------- | ---------- | ---- | --------------------------- | ------------------------------ |
| check_brackets                          | 栈          | *    | *`list`,`pop(index)` list实现栈，因为操作在末尾       | *`util.Stack`,`peek()`         |
| tree_height                             |            | *    | *`嵌套list`,`enumerate`,`map` | *`嵌套ArrayList`,`get()`,`add()` |
| process_packages                        | 队列           | *    | *`sys.stdin.readline()` `collections.deque`实现队列，因为要在队首操作，list的效率会低一些   | *`ArrayList`,`remove()`        |

## week2 Priority Queues and Disjoint Sets

* 建堆，堆排序，优先队列
* 并查集，路径压缩，合并

| Assignments                             | Tags       | C++  | Python                        | Java                                             |
| --------------------------------------- | ---------- | ---- | ----------------------------- | ------------------------------------------------ |
| make_heap                               | 建堆，调整      | *    | * 交换两个数                       | *交换两个数，数组作为参数                                    |
| job_queue                               | 优先级队列      | *    | *`heapq`,`heappush`,`heappop` 重载运算符__lt__ | *`Queue,PriorityQueue,Comparator` ,`add,poll,pop` |
| merging_tables                          |            |      |                               |                                                  |

## week3 Hash Tables

* 整数哈希，字符串哈希
* 模式匹配：`Rabin-Karp`算法
  * `PolyHash`
  * `PreComputeHash`
  * `p>>|T||P|,x=random(1,p)`
  * 避免与无符号数作减法溢出，避免求余时结果为负
* 分布式哈希表，一致性哈希

## week4 Binary Search Trees

* 二叉树的遍历，先序，中序，后序的递归和迭代形式，层序遍历
* 平衡二叉树
* Splay伸展树 查找，调整，添加，删除