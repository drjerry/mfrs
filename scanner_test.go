package mfrs

import (
	"strings"
	"testing"
)

func TestScannerCorrectness(t *testing.T) {
	rows := []int{0, 1}
	cols := []int{1, 0}
	vals := []float32{1.1, 2.1}
	txt := "0 1 1.1\n1 0 2.1"

	reader := strings.NewReader(txt)
	scanner := NewScanner(reader)

	for i := 0; i < 2; i++ {
		scanner.Scan()
		row, col, val := scanner.Record()
		if row != rows[i] || col != cols[i] || val != vals[i] {
			t.Errorf("iter %d => (%d, %d, %g)", i, row, col, val)
		}
	}

	if scanner.Scan() {
		t.Error("expected end-state not met")
	}
}
