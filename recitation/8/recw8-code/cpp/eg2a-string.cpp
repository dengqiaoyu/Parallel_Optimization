// Comparison with C strings
// Does this code work?
#include <iostream>
#include <stdio.h>
#include <string.h>

char *stringify(char *s) {
    char buf[10];
    strcpy(buf, s);
    return buf;
}

int main(int arg, char *argv[]) {
    char buf[10];
    char *strings[5];
    for (int i = 0; i < 5; i++) {
	sprintf(buf, "i=%d", i);
	strings[i] = stringify(buf);
    }
    for (int i = 0; i < 5; i++) {
	std::cout << "String " << i << ":"<< strings[i] << std::endl;
    }
    return 0;
}
