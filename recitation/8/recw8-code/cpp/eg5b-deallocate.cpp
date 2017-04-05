// Using destructors to deallocate linked list elements
// Demonstration of overly aggressive deallocation
// Avoids excessive deallocation
#include <iostream>

class ListEle {
private:
    int val;
    ListEle *next;

public:
    ListEle(int v, ListEle *np)
    { val = v; next = np; }

    ListEle *getNext() { return next; }

    int getValue() { return val; }

    void unlink() { next = NULL; }

    ~ListEle() {
	std::cout << "Destroying element with value " << val << std::endl;
	if (next)
	    delete next;
    }
};

class List {
private:
    ListEle *head;

public:
    List() { head = NULL; }

    void insert(int v) { head = new ListEle(v, head); }

    int front() {
	if (head) return head->getValue();
	else      return -1;
    }

    void pop() {
	if (head) {
	    ListEle *save = head;
	    head = head->getNext();
	    save->unlink();
	    delete save;
	}
    }

    bool isEmpty() { return head == NULL; }

    ~List() { if (head) delete head; }
};

int main(int arg, char *argv[]) {
    List ls;
    for (int i = 0; i < 5; i++) {
	ls.insert(i);
    }

    while (!ls.isEmpty()) {
	int v = ls.front();
	std::cout << "Popped value " << v << std::endl;
	ls.pop();
    }
    return 0;
}
