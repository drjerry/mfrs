package linalg

import "fmt"

type Vector []float32

// NewVector returns a vector allocated with the given size.
func NewVector(size int) Vector {
	return make([]float32, size)
}

// Matrix holds a row-major matrix representation.
type Matrix struct {
	nrow, ncol int
	vals       []float32
}

/* MatrixView wraps the `values` array in a row-major matrix representation.
It raises an error if the declared nrow-by-ncol dimension of the matrix does
not match the size of the underlying array.
*/
func MatrixView(nrow, ncol int, values []float32) (Matrix, error) {
	if nrow*ncol != len(values) {
		return Matrix{}, fmt.Errorf("matrix shape %d,%d invalid", nrow, ncol)
	}
	return Matrix{nrow, ncol, values}, nil
}

// RowView maps the provided Vector onto the i-th row.
func (m Matrix) RowView(i int, view *Vector) {
	(*view) = m.vals[i*m.ncol : (i+1)*m.ncol]
}

// Copy moves the `src` vector into the underlying vector.
func (v Vector) Copy(src Vector) {
	copy(v, src)
}
