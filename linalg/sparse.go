package linalg

import "fmt"

/* SparseMatrix is a simplified sparse matrix representation.

By design it has no get/set accessors for individual members, and instead
is an "iterative" container. Data can only be appended to the structure
as (row, col, value) triples, and the data must be sorted lexicographically
by (row, col). (This supports the underlying compression schema.) Likewise,
data is read out via an Iterator interface, and is returned in the same
sequential order.
*/

type SparseMatrix struct {
	nrow, ncol int
	rptr       []int
	cind       []int
	vals       []float32
	row        int // internal state for Add method
}

type SparseIterator struct {
	m        *SparseMatrix
	row, pos int
}

// NewSparseMatrix initializes an nrow-by-ncol matrix with no data.
func NewSparseMatrix(nrow, ncol int) SparseMatrix {
	return SparseMatrix{
		nrow: nrow,
		ncol: ncol,
		rptr: make([]int, nrow+1),
	}
}

// Shape returns the (nrow, ncol) pair.
func (m SparseMatrix) Shape() (int, int) {
	return m.nrow, m.ncol
}

// NNZ returns the number of non-zero entries.
func (m SparseMatrix) NNZ() int {
	return len(m.cind)
}

/* Add appends an element (row, col, value) to the matrix. An error is raised
if (row, col) is out of sequence from the previous entry added.
*/
func (m *SparseMatrix) Add(row, col int, value float32) error {
	n := len(m.cind)
	if n > 0 {
		if row < m.row || (row == m.row && col < m.cind[n-1]) {
			return fmt.Errorf("entry %d,%d out of sequence", row, col)
		}
	}
	m.cind = append(m.cind, col)
	m.vals = append(m.vals, value)
	m.rptr[row+1] = n + 1
	m.row = row
	// backfill skipped rptr slots to ensure monotonicity
	for row > 0 && m.rptr[row] < m.rptr[row-1] {
		m.rptr[row] = m.rptr[row-1]
		row--
	}
	return nil
}

func (m SparseMatrix) Iterator() SparseIterator {
	return SparseIterator{m: &m}
}

func (it SparseIterator) HasNext() bool {
	return it.pos < len(it.m.cind)
}

func (it *SparseIterator) Next() (int, int, float32) {
	col := it.m.cind[it.pos]
	val := it.m.vals[it.pos]
	for it.m.rptr[it.row+1] <= it.pos {
		it.row++
	}
	it.pos++
	return it.row, col, val
}
