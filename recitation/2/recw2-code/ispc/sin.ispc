#define MAXTERMS 30

// Conversion of original version of sin function to ISPC
// (Note how this differs from ISPC code shown in lecture.)
export void sinx_reference(uniform int N, uniform int terms,
			   uniform float x[],
			   uniform float result[])
{
    foreach (i=0 ... N) {
	float value = x[i];
	float numer = x[i]*x[i]*x[i];
	uniform int denom = 6; // 3!
	uniform int sign = -1;
	for (uniform int j=1; j<=terms; j++) {
	    value += sign * numer / denom;
	    numer *= x[i] * x[i];
	    denom *= (2*j+2) * (2*j+3);
	    sign *= -1;
	}

	result[i] = value;
    }
}


// Rewrite the code for your best performing regular 
export void sinx_best(uniform int N, uniform int terms,
		      uniform float x[],
       	              uniform float result[])
{

    // Your code here

}

// Rewrite the reference code to be suitable for a task
//   with npt elements per thread

// You should correctly handle the case where N is not
//   divisible by the number of threads

task void sinx_reference_t(uniform int N, uniform int terms,
			   uniform float x[],
     	                   uniform float result[],
			   uniform int npt)
{

    // Your code here

}


// Compute with two threads running the reference solution

// You should correctly handle the case where N is not
//   divisible by the number of threads
export void sinx_reference_task2(uniform int N,
				 uniform int terms,
       	    			 uniform float x[],
				 uniform float result[]) {
       uniform int taskCount = 2;

    // Your code here

}



// Rewrite your best version to be suitable for a task
//   with npt elements per thread

// You should correctly handle the case where N is not
//   divisible  by the number of threads

task void sinx_best_t(uniform int N,
		      uniform int terms,
		      uniform float x[],
     	              uniform float result[],
		      uniform int npt)
{

    // Your code here

}

// Compute with two threads running your best solution
// You should correctly handle the case where N is not
//  divisible by the number of threads
export void sinx_best_task2(uniform int N,
			    uniform int terms,
       	    	            uniform float x[],
			    uniform float result[]) {
       uniform int taskCount = 2;

    // Your code here

}
