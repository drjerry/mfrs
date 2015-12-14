package mfrs

import (
	"github.com/golang/protobuf/proto"
	"io/ioutil"
	"log"
	"os"
)

// NewModel returns an allocated Model with specified dimensions.
func NewModel(ldim, nrow, ncol int) Model {
	return Model{
		int32(ldim), 
		int32(nrow), 
		int32(ncol),
		make([]float32, nrow * ldim),
		make([]float32, ncol * ldim),
	}
}

// SaveModel writes the Model to the specified file.
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

// LoadModel deserialized a Model from the specified file.
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
