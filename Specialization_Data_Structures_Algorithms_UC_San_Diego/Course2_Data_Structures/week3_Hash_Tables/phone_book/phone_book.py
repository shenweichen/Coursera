# python3
import sys
class Query:
    def __init__(self, query):
        self.type = query[0]
        self.number = int(query[1])
        if self.type == 'add':
            self.name = query[2]

def read_queries():
    n = int(input())
    return [Query(sys.stdin.readline().split()) for i in range(n)]

def write_responses(result):
    print('\n'.join(result))

def process_queries(queries):
    result = []
    # Keep list of all existing (i.e. not deleted yet) contacts.
    contacts = [[] for _ in range(0,100000)]
    for cur_query in queries:
        hash_ = cur_query.number%100000
        if cur_query.type == 'add':
            was_found = False
            # if we already have contact with such number,
            # we should rewrite contact's name
            for contact in contacts[hash_]:
                if contact.number == cur_query.number:
                    contact.name = cur_query.name
                    was_found = True
                    break
            if(not was_found): # otherwise, just add it
                contacts[hash_].append(cur_query)
        elif cur_query.type == 'del':
            for contact in contacts[hash_]:
                if contact.number == cur_query.number:
                    contacts[hash_].remove(contact)
                    break
        else:
            response = 'not found'
            for contact in contacts[hash_]:
                if contact.number == cur_query.number:
                    response = contact.name
                    break
            result.append(response)
    return result

if __name__ == '__main__':
    write_responses(process_queries(read_queries()))

