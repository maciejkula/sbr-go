package main

import (
	"fmt"
	"math/rand"

	sbr "github.com/maciejkula/sbr-go"
)

func main() {
	// Load the data.
	data, err := sbr.GetMovielens()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded movielens data: %v users and %v items for a total of %v interactions\n",
		data.NumUsers(), data.NumItems(), data.Len())

	// Split into test and train.
	rng := rand.New(rand.NewSource(42))
	train, test := sbr.TrainTestSplit(data, 0.2, rng)
	fmt.Printf("Train len %v, test len %v\n", train.Len(), test.Len())

	// Instantiate the model.
	model := sbr.NewImplicitLSTMModel(train.NumItems())

	// Set the hyperparameters.
	model.ItemEmbeddingDim = 32
	model.LearningRate = 0.16
	model.L2Penalty = 0.0004
	model.NumEpochs = 10
	model.NumThreads = 1

	// Set random seed
	var randomSeed [16]byte
	for idx := range randomSeed {
		randomSeed[idx] = 42
	}
	model.RandomSeed = randomSeed

	// Fit the model.
	fmt.Printf("Fitting the model...\n")
	loss, err := model.Fit(&train)
	if err != nil {
		panic(err)
	}

	// And evaluate.
	fmt.Printf("Evaluating the model...\n")
	mrr, err := model.MRRScore(&test)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loss %v, MRR: %v\n", loss, mrr)
}
