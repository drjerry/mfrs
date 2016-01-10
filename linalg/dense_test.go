package linalg

import "testing"

func TestMatrixView(t *testing.T) {
	nrow, ncol := 2, 3
	vals := make([]float32, 6)

	m, _ := MatrixView(nrow, ncol, vals)
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

	m, _ := MatrixView(nrow, ncol, vals)
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
	ss, ds, ts := NewVector(2), NewVector(2), NewVector(2)
	ts[0], ts[1] = 1, 2 // "truth" vector

	ss[0], ss[1] = ts[0], ts[1]
	ds.Copy(ss)
	for i, v := range ds {
		if ts[i] != v {
			t.Errorf("Vector.Copy => %v != %v", ts, ds)
		}
	}
}
