package mfrs

import (
	"fmt"
	"os"
	"testing"
)

func floatArrayEqual(xs, ys []float32) error {
	if len(xs) != len(ys) {
		return fmt.Errorf("%v != %v", xs, ys)
	}
	for i, x := range xs {
		if x != (ys)[i] {
			return fmt.Errorf("%v != %v", xs, ys)
		}
	}
	return nil
}

func TestModelSerialization(t *testing.T) {
	testName := "/tmp/model.test"
	src := NewModel(1, 2, 3)
	src.Pwts = []float32{1, 2}
	src.Qwts = []float32{1, 2, 3}
	src.Pbias = []float32{1.5, 2.5}
	src.Qbias = []float32{1.5, 2.5, 3.5}

	if err := SaveModel(&src, testName); err != nil {
		t.Error(err)
	}

	dest, err := LoadModel(testName)
	if err != nil {
		t.Error(err)
	}
	os.Remove(testName)

	if err = floatArrayEqual(src.Pwts, dest.Pwts); err != nil {
		t.Error(err)
	}

	if err = floatArrayEqual(src.Qwts, dest.Qwts); err != nil {
		t.Error(err)
	}
}
