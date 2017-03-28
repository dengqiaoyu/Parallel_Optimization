// Demonstration of C++ strings
// Alternative way to initialize string
#include <string>
#include <iostream>
#include <stdio.h>

void stringify(char *s, std::string &ss) {
    ss = s;
}

int main(int arg, char *argv[]) {
    char buf[10];
    std::string strings[5];
    for (int i = 0; i < 5; i++) {
	sprintf(buf, "i=%d", i);
	stringify(buf, strings[i]);
    }
    for (int i = 0; i < 5; i++) {
	std::cout << "String " << i << ":"<< strings[i] << std::endl;
    }
    return 0;
}
