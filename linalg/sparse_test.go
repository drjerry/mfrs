package linalg

import "testing"

func TestFindCol(t *testing.T) {
	fxt := new(CRSMatrix)
	fxt.CInd = []int{1, 2, 4}

	if res := fxt.findCol(0, 3, 1); res != 0 {
		t.Errorf("findCol(..,1) => %d", res)
	}
	if res := fxt.findCol(0, 3, 2); res != 1 {
		t.Errorf("findCol(..,2) => %d", res)
	}
	if res := fxt.findCol(0, 3, 4); res != 2 {
		t.Errorf("findCol(..,4) => %d", res)
	}
	if res := fxt.findCol(0, 3, 3); res != -1 {
		t.Errorf("findCol(..,3) => %d", res)
	}
}

type entry struct {
	i, j int
	v    float32
}

func (e *entry) isEqual(i, j int, v float32) bool {
	return (e.i == i && e.j == j && e.v == v)
}

func TestSmokeTestBuilderIter(t *testing.T) {
	fxt := NewCRSMatrix(3, 3, 3)
	data := []entry{
		entry{0, 1, float32(1)},
		entry{0, 2, float32(2)},
		entry{2, 0, float32(3)},
	}

	bdr := fxt.Builder()
	for _, x := range data {
		if err := bdr.Put(x.i, x.j, x.v); err != nil {
			t.Errorf("%v => %s", x, err)
		}
	}
	if err := bdr.Put(2, 1, float32(0)); err == nil {
		t.Errorf("expected CSRBuilder.Put error state not met")
	}

	it := fxt.Iterator()
	for _, x := range data {
		i, j, v := it.Next()
		if !x.isEqual(i, j, v) {
			t.Error("%v != (%d, %d, %g)", x, i, j, v)
		}
	}
	if it.HasNext() {
		t.Errorf("expected CSRIterator.Next end-state not met")
	}
}
