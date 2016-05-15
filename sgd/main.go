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
	Sigma     float32
}

func main() {
	log.SetFlags(log.Lmicroseconds)
	args, err := parseArgs()
	if err != nil {
		flag.PrintDefaults()
		log.Fatal(err)
	}

	model := mfrs.NewModel(args.Ldim, args.Nrow, args.Ncol)
	rand.Seed(args.Seed)
	model.Randomize(args.Sigma)

	infile, err := os.Open(args.Infile)
	if err != nil {
		log.Fatal(err)
	}

	scanner := mfrs.NewScanner(infile)
	solver := NewSGDSolver(&model, args)

	var count int
	var mse float32
	var data mfrs.Ratings
	log.Printf("reading %s", infile.Name())
	for scanner.Scan() {
		row, col, val := scanner.Record()
		data.Add(row, col, val)
		delta := solver.Update(row, col, val)
		count++
		mse += (delta - mse) / float32(count)
	}
	if scanner.Err() != nil {
		log.Fatal(scanner.Err())
	}

	log.Printf("iter 1, RMSE %.6g\n", math.Sqrt(float64(mse)))

	for iter := 2; iter <= args.MaxIter; iter++ {
		mse = 0
		for n := 0; n < len(data); n++ {
			delta := solver.Update(data[n].Row, data[n].Col, data[n].Val)
			mse += (delta - mse) / float32(n+1)
		}
		log.Printf("iter %d, RMSE %.6g\n", iter, math.Sqrt(float64(mse)))
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
	sigma := flag.Float64("sigma", 0.7e-1, "std dev for initial weights")
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
	args.Sigma = float32(*sigma)
	return args, nil
}
