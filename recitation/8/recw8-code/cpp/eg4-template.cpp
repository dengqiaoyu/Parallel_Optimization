// Using template to create generic linked list
#include <iostream>

template <class T>
class ListEle {
private:
    T val;
    ListEle *next;
public:
    
    ListEle(T v, ListEle *np) 
    { val = v; next = np; }

    ListEle *getNext() { return next; }

    T getValue() { return val; }
};

template <class T>
class List {
private:
    ListEle<T> *head;
public:
    List() { head = NULL; }

    void insert(T v) { head = new ListEle<T>(v, head); }

    T front() {	return head->getValue(); }

    void pop() {
	if (head) {
	    ListEle<T> *save = head;
	    head = head->getNext();
	    delete save;
	}
    }

    bool isEmpty() { return head == NULL; }
};

int main(int arg, char *argv[]) {
    List<int> ls;
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
