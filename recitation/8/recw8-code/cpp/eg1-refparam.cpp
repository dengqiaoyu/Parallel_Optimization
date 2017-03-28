// Demonstration of reference parameters
#include <iostream>

// C style
int pincr(int *xp) {
    int r = *xp;
    (*xp)++;
    return r;
}

// C++ style
int rincr(int &x) {
    int r = x;
    x++;
    return r;
}

int main(int argc, char *argv[]) {
    int x = 1;
    int y = pincr(&x);
    int z = rincr(x);
    std::cout << "x=" << x << ", y=" << y << ", z=" << z << std::endl;
    return 0;
}

