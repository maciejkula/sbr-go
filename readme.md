# sbr-go
[![Build Status](https://travis-ci.org/maciejkula/sbr-go.svg?branch=master)](https://travis-ci.org/maciejkula/sbr-go)
[![Godoc](https://img.shields.io/badge/godoc-reference-5272B4.svg?style=flat-square)](https://godoc.org/github.com/maciejkula/sbr-go)

A recommender system package for Go.

Sbr implements state-of-the-art sequence-based models, using the history of what a user has liked to suggest new items. As a result, it makes accurate prediction that can be updated in real-time in response to user actions without model re-training.

## Usage
You can fit a model on the Movielens 100K dataset in about 10 seconds using the following code:
```go
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
```

## Installation requirements

You will need the Rust compiler. You can install Rust from [here](https://www.rust-lang.org/en-US/install.html) by running
```
curl https://sh.rustup.rs -sSf | sh
```

## Installation
Run
```
go get github.com/maciejkula/sbr-go
```
followed by
```
make
```
in the installation directory. This wil compile the package's native dependencies.
The resulting libraries (in `./lib`) must be available and in the dylib loading
path at runtime.

You may have to set
```
export CGO_LDFLAGS_ALLOW="-Wl*"
```
depending on your Go version.
