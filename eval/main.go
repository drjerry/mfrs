package main

import (
	"flag"
	"fmt"
	"github.com/drjerry/mfrs"
	"log"
	"os"
)

type Args struct {
	Infile    string
	ModelFile string
}

func main() {
	args, err := parseArgs()
	if err != nil {
		flag.PrintDefaults()
		log.Fatal(err)
	}

	model, err := mfrs.LoadModel(args.ModelFile)
	if err != nil {
		log.Fatal(err)
	}

	infile, err := os.Open(args.Infile)
	if err != nil {
		log.Fatal(err)
	}

	scanner := mfrs.NewScanner(infile)
	for scanner.Scan() {
		row, col, val := scanner.Record()
		pred := model.Predict(row, col)
		fmt.Printf("%d %d %g %g\n", row, col, pred, val)
	}
	if scanner.Err() != nil {
		log.Fatal(scanner.Err())
	}
}

func parseArgs() (*Args, error) {
	args := new(Args)
	flag.StringVar(&args.Infile, "in", "", "sparse matrix data file")
	flag.StringVar(&args.ModelFile, "model", "", "input file for model")
	flag.Parse()

	if args.Infile == "" {
		return nil, fmt.Errorf("'in' argument missing")
	}
	if args.ModelFile == "" {
		return nil, fmt.Errorf("'model' argument missing")
	}
	return args, nil
}
