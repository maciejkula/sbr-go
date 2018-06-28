# sbr-go
[![Build Status](https://travis-ci.org/maciejkula/sbr-go.svg?branch=master)](https://travis-ci.org/maciejkula/sbr-go)
[![Godoc](https://img.shields.io/badge/godoc-reference-5272B4.svg?style=flat-square)](https://godoc.org/github.com/maciejkula/sbr-go)

A recommender system package for Go.

Sbr implements state-of-the-art sequence-based models, using the history of what a user has liked to suggest new items. As a result, it makes accurate predictions that can be updated in real-time in response to user actions without model re-training.

Sbr implements cutting-edge sequence-based recommenders: for every user, we examine what
they have interacted up to now to predict what they are going to consume next.

Implemented models:
- LSTM: a model that uses an LSTM network over the sequence of a user's interaction
        to predict their next action;
- EWMA: a model that uses a simpler exponentially-weighted average of past actions
        to predict future interactions.

Which model performs the best will depend on your dataset. The EWMA model is much
quicker to fit, and will probably be a good starting point.

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
## Installation
Run
```
go get github.com/maciejkula/sbr-go
```
followed by
```
make
```
in the installation directory. This will download the package's native dependencies. On both OSX and Linux, the resulting binaries are fully statically linked, and you can deploy them like any other Go binary.

If you prefer to build the dependencies from source, run `make source` instead.
