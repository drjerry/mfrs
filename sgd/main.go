package main

import (
	"flag"
	"fmt"
	"github.com/drjerry/mfrs"
	"log"
	"math"
	"math/rand"
	"os"
)

type Args struct {
	Ldim      int
	Nrow      int
	Ncol      int
	Infile    string
	ModelFile string
	Rate      float32
	Lambda    float32
	MaxIter   int
	Seed      int64
}

func main() {
	args, err := parseArgs()
	if err != nil {
		log.Fatal(err)
	}

	model := mfrs.NewModel(args.Ldim, args.Nrow, args.Ncol)
	rand.Seed(args.Seed)
	randomize(model.Pvals)
	randomize(model.Qvals)
	log.Print("initialized model")

	infile, err := os.Open(args.Infile)
	if err != nil {
		log.Fatal(err)
	}

	var data mfrs.Ratings
	scanner := mfrs.NewScanner(infile)
	for scanner.Scan() {
		row, col, val := scanner.Record()
		data.Add(row, col, val)
	}
	if scanner.Err() != nil {
		log.Fatal(scanner.Err())
	}

	for i := 0; i < args.MaxIter; i++ {
		mse := sgd(args, data, &model)
		log.Printf("iter %d, RMSE %.6g\n", i, math.Sqrt(float64(mse)))
	}

	mfrs.SaveModel(&model, args.ModelFile)
}

func parseArgs() (*Args, error) {
	args := new(Args)
	flag.IntVar(&args.Ldim, "ldim", 0, "latent dimension")
	flag.IntVar(&args.Nrow, "nrow", 0, "user-space dimension")
	flag.IntVar(&args.Ncol, "ncol", 0, "item-space dimension")
	flag.StringVar(&args.Infile, "in", "", "sparse matrix data file")
	flag.StringVar(&args.ModelFile, "fit", "", "target file for model")
	flag.IntVar(&args.MaxIter, "epochs", 10, "maximum iteration count")
	flag.Int64Var(&args.Seed, "seed", 2908, "RNG seed")
	rate := flag.Float64("rate", 1e-2, "learning rate")
	lambda := flag.Float64("lambda", 1e-3, "regularization parameter")
	flag.Parse()

	required := []string{"ldim", "nrow", "ncol", "in", "fit"}
	for _, name := range required {
		arg := flag.Lookup(name)
		if arg.DefValue == arg.Value.String() {
			return nil, fmt.Errorf("argument '%s' missing", name)
		}
	}

	args.Rate = float32(*rate)
	args.Lambda = float32(*lambda)
	return args, nil
}

func randomize(xs []float32) {
	for i := 0; i < len(xs); i++ {
		xs[i] = rand.Float32()
	}
}
