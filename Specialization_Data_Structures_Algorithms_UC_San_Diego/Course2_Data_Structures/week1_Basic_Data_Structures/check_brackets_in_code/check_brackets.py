# python3

import sys

class Bracket:
    def __init__(self, bracket_type, position):
        self.bracket_type = bracket_type
        self.position = position

    def Match(self, c):
        if self.bracket_type == '[' and c == ']':
            return True
        if self.bracket_type == '{' and c == '}':
            return True
        if self.bracket_type == '(' and c == ')':
            return True
        return False

if __name__ == "__main__":
    text = sys.stdin.read()
    ans = -1
    opening_brackets_stack = []
    for i, next in enumerate(text):
        if next == '(' or next == '[' or next == '{':
            # Process opening bracket, write your code here
            opening_brackets_stack.append(Bracket(next,i))

        if next == ')' or next == ']' or next == '}':
            # Process closing bracket, write your code here
            if(len(opening_brackets_stack)==0 or not opening_brackets_stack[-1].Match(next)):
                ans = i +1
                break
            opening_brackets_stack.pop()
    # Printing answer, write your code here
    if(ans == -1 and len(opening_brackets_stack)==0):
        print("Success")
    elif(ans == -1):
        print(opening_brackets_stack[-1].position+1)
    else:
        print(ans)