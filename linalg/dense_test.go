package linalg

import "testing"

func TestMatrixView(t *testing.T) {
	nrow, ncol := 2, 3
	vals := make([]float32, 6)

	m := MatrixView(nrow, ncol, vals)
	for i := 0; i < (m.nrow * m.ncol); i++ {
		m.vals[i] = float32(i + 1)
	}

	for i, v := range vals {
		if v != float32(i+1) {
			t.Errorf("vals => %v", vals)
		}
	}
}

func TestRowView(t *testing.T) {
	nrow, ncol := 2, 3
	vals := make([]float32, 6)
	var vec Vector

	m := MatrixView(nrow, ncol, vals)
	m.RowView(1, &vec)
	for i := 0; i < len(vec); i++ {
		vec[i] = float32(i + 1)
	}

	for j := 0; j < 3; j++ {
		if vals[j+3] != float32(j+1) {
			t.Errorf("vals => %v", vals)
		}
	}
}

func TestVectorCopy(t *testing.T) {
	ts := Vector([]float32{1, 2})  // "truth" vector
	ss := Vector([]float32{1, 2})  // "source"
	ds := NewVector(2)  // "dest"

	ds.Copy(ss)
	for i, v := range ds {
		if ts[i] != v {
			t.Errorf("Vector.Copy => %v != %v", ts, ds)
		}
	}
}
