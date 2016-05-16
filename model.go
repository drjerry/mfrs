package mfrs

import (
	"github.com/drjerry/mfrs/linalg"
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

// Initialize sets the initial weights for the model using:
// (1) constant `bias` value for the bias vectors,
// (2) randomized weight vectors, Normal dist with std deviation `sigma`.
func (model *Model) Initialize(bias, sigma float32) {
	log.Print("initializing model")
	randomize(sigma, model.Pvals)
	randomize(sigma, model.Qvals)
	assign(sigma, model.Pbias)
	assign(sigma, model.Qbias)
}

// Predict evaluates the model for the predicted rating at <row, col>.
func (model *Model) Predict(row, col int) float32 {
	stride := int(model.Ldim)
	pOffset := row * stride
	qOffset := col * stride
	pRow := linalg.Vector(model.Pvals[pOffset: pOffset+stride])
	qRow := linalg.Vector(model.Qvals[qOffset: qOffset+stride])
	return model.Pbias[row] + model.Qbias[col] + linalg.Vdot(pRow, qRow)
}

func randomize(sigma float32, xs []float32) {
	for i := 0; i < len(xs); i++ {
		xs[i] = float32(rand.NormFloat64()) * sigma
	}
}

func assign(bias float32, xs []float32) {
	for i := 0; i < len(xs); i++ {
		xs[i] = bias
	}
}
