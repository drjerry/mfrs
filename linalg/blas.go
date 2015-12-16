package linalg

// #include <cblas.h>
import "C"

// Vdot returns the inner product: x^T y
func Vdot(x, y Vector) float32 {
	return float32(C.cblas_sdot(C.int(len(x)), (*C.float)(&x[0]), 1, (*C.float)(&y[0]), 1))
}

// Vnrm2 Returns the Euclidean norm: \sqrt{ x^T x }
func Vnrm2(x Vector) float32 {
	return float32(C.cblas_snrm2(C.int(len(x)), (*C.float)(&x[0]), 1))
}

// Vaxpy updates y via translation: y <- \alpha x + y
func Vaxpy(alpha float32, x, y Vector) {
	C.cblas_saxpy(C.int(len(x)), C.float(alpha), (*C.float)(&x[0]), 1, (*C.float)(&y[0]), 1)
}

// Vscal rescales the vector:  x <- \alpha x
func Vscal(alpha float32, x Vector) {
	C.cblas_sscal(C.int(len(x)), C.float(alpha), (*C.float)(&x[0]), 1)
}
