package mfrs

import (
	"github.com/golang/protobuf/proto"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
)

// NewModel returns an allocated Model with specified dimensions.
func NewModel(ldim, nrow, ncol int) Model {
	return Model{
		int32(ldim),
		int32(nrow),
		int32(ncol),
		make([]float32, nrow*ldim),
		make([]float32, ncol*ldim),
		make([]float32, nrow),
		make([]float32, ncol),
	}
}

// SaveModel serializes the Model to the specified file.
func SaveModel(model *Model, filename string) error {
	data, err := proto.Marshal(model)
	if err != nil {
		return err
	}

	writer, err := os.Create(filename)
	if err != nil {
		return err
	}

	log.Printf("writing %s", filename)
	_, err = writer.Write(data)
	if err != nil {
		return err
	}
	writer.Close()
	return nil
}

// LoadModel deserializes a Model from the specified file.
func LoadModel(filename string) (*Model, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	log.Printf("reading %s", filename)
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}
	reader.Close()

	model := new(Model)
	err = proto.Unmarshal(data, model)
	if err != nil {
		return nil, err
	}
	return model, nil
}

// Randomize initializes the weights as IID Gaussian with specified std dev.
func (model *Model) Randomize(sigma float32) {
	randomize(sigma, model.Pvals)
	randomize(sigma, model.Qvals)
	randomize(sigma, model.Pbias)
	randomize(sigma, model.Qbias)
	log.Print("randomized weights")
}

func randomize(sigma float32, xs []float32) {
	for i := 0; i < len(xs); i++ {
		xs[i] = float32(rand.NormFloat64()) * sigma
	}
}
