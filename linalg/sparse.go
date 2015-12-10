package linalg

import "fmt"

type CRSMatrix struct {
	NRow int
	NCol int
	NNZ  int
	RPtr []int
	CInd []int
	Val  []float32
}

func NewCRSMatrix(nrow, ncol, nnz int) *CRSMatrix {
	m := new(CRSMatrix)
	m.NRow, m.NCol, m.NNZ = nrow, ncol, nnz
	m.RPtr = make([]int, nrow+1)
	m.CInd = make([]int, nnz)
	m.Val = make([]float32, nnz)
	return m
}

// Finds the element `j` in the segment [CInd[il], CInd[iu]). Returns the
// index in CInd if the element is present, and -1 otherwise.
func (m *CRSMatrix) findCol(il, iu, j int) int {
	for m.CInd[il] < j {
		if iu-il <= 1 {
			return -1
		}
		im := il + (iu-il)/2
		if m.CInd[im] <= j {
			il = im
		} else {
			iu = im
		}
	}
	return il
}

func (m *CRSMatrix) Get(i, j int) (float32, error) {
	k := m.findCol(m.RPtr[i], m.RPtr[i+1], j)
	if k == -1 {
		return float32(0), nil
	}
	return m.Val[k], nil
}

func (m *CRSMatrix) Set(i, j int, v float32) error {
	k := m.findCol(m.RPtr[i], m.RPtr[i+1], j)
	if k == -1 {
		return fmt.Errorf("index (%d, %d) invalid", i, j)
	}
	m.Val[k] = v
	return nil
}

type CRSBuilder struct {
	m   *CRSMatrix
	pos int
}

func (m *CRSMatrix) Builder() *CRSBuilder {
	b := new(CRSBuilder)
	b.m = m
	return b
}

func (b *CRSBuilder) Put(i, j int, v float32) error {
	if b.pos == b.m.NNZ {
		return fmt.Errorf("matrix capacity filled")
	}
	b.m.CInd[b.pos] = j
	b.m.Val[b.pos] = v
	b.pos++
	b.m.RPtr[i+1] = b.pos
	return nil
}

type CRSIterator struct {
	m   *CRSMatrix
	row int
	pos int
}

func (m *CRSMatrix) Iterator() *CRSIterator {
	it := new(CRSIterator)
	it.m = m
	return it
}

func (it *CRSIterator) HasNext() bool {
	return it.pos != it.m.NNZ
}

func (it *CRSIterator) Next() (int, int, float32) {
	col := it.m.CInd[it.pos]
	val := it.m.Val[it.pos]
	for it.m.RPtr[it.row+1] <= it.pos {
		it.row++
	}
	it.pos++
	return it.row, col, val
}
