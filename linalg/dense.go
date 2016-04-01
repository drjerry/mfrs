package linalg

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

// MatrixView wraps the `values` array in a row-major matrix representation.
func MatrixView(nrow, ncol int, values []float32) Matrix {
	if nrow*ncol != len(values) {
		panic("MatrixView: dimension mismatch")
	}
	return Matrix{nrow, ncol, values}
}

// RowView maps the provided Vector onto the i-th row.
func (m Matrix) RowView(i int, view *Vector) {
	(*view) = m.vals[i*m.ncol : (i+1)*m.ncol]
}

// Copy moves the `src` vector into the underlying vector.
func (v Vector) Copy(src Vector) {
	copy(v, src)
}
