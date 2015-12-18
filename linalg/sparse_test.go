package linalg

import (
	"fmt"
	"testing"
)

type entry struct {
	row, col int
	val      float32
}

func (e entry) isEqual(row, col int, val float32) bool {
	return (e.row == row && e.col == col && e.val == val)
}

func intSliceTest(xs, ys []int) error {
	if len(xs) != len(ys) {
		return fmt.Errorf("length mismatch: %v != %v", xs, ys)
	}
	for i, x := range xs {
		if x != ys[i] {
			return fmt.Errorf("index %d: %v =! %v", i, xs, ys)
		}
	}
	return nil
}

func floatSliceTest(xs, ys []float32) error {
	if len(xs) != len(ys) {
		return fmt.Errorf("length mismatch: %v != %v", xs, ys)
	}
	for i, x := range xs {
		if x != ys[i] {
			return fmt.Errorf("index %d: %v =! %v", i, xs, ys)
		}
	}
	return nil
}

func TestSparseBuilder(t *testing.T) {
	//	[0  1  2]
	//	[0  0  0]
	//	[3  0  0]
	data := []entry{
		entry{0, 1, 1.0},
		entry{0, 2, 2.0},
		entry{2, 0, 3.0},
	}

	// expected Rptr and Cind values
	rptr := []int{0, 2, 2, 3}
	cind := []int{1, 2, 0}
	vals := []float32{1, 2, 3}

	sm := NewSparseMatrix(3, 3)
	for _, e := range data {
		if err := sm.Add(e.row, e.col, e.val); err != nil {
			t.Errorf("build error => %v", err)
		}
	}

	if err := intSliceTest(sm.rptr, rptr); err != nil {
		t.Error(err)
	}
	if err := intSliceTest(sm.cind, cind); err != nil {
		t.Error(err)
	}
	if err := floatSliceTest(sm.vals, vals); err != nil {
		t.Error(err)
	}
}

func TestSparseIterators(t *testing.T) {
	sm := NewSparseMatrix(3, 3)
	sm.rptr = []int{0, 2, 2, 3}
	sm.cind = []int{1, 2, 0}
	sm.vals = []float32{1, 2, 3}

	expected := []entry{
		entry{0, 1, 1.0},
		entry{0, 2, 2.0},
		entry{2, 0, 3.0},
	}

	iter := sm.Iterator()
	for _, x := range expected {
		row, col, val := iter.Next()
		if !x.isEqual(row, col, val) {
			t.Errorf("%v != (%d, %d, %g)", x, row, col, val)
		}
	}
	if iter.HasNext() {
		t.Errorf("invalid SparseIterator end-state")
	}
}
