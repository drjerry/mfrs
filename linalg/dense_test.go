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
