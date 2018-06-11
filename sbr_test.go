package sbr

import (
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"
)

func expectedMRR() float32 {
	switch simd := os.Getenv("MKL_CBWR"); simd {
	case "SSE4_1":
		return 0.082
	case "AVX":
		return 0.083
	default:
		return 0.07
	}
}

func TestMovielens100K(t *testing.T) {
	data, err := GetMovielens()
	if err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(42))
	train, test := TrainTestSplit(data, 0.2, rng)

	fmt.Printf("Train len %v, test len %v\n", train.Len(), test.Len())

	fmt.Println("Building model...")
	model := NewImplicitLSTMModel(data.NumItems())

	// Set the hyperparameters.
	model.MaxSequenceLength = 32
	model.ItemEmbeddingDim = 32
	model.LearningRate = 0.16
	model.L2Penalty = 0.0004
	model.NumEpochs = 15
	model.NumThreads = 1
	model.Loss = Hinge
	model.Optimizer = Adagrad

	// Set random seed
	var randomSeed [16]byte
	for idx := range randomSeed {
		randomSeed[idx] = 42
	}
	model.RandomSeed = randomSeed

	fmt.Println("Fitting...")
	start := time.Now()
	loss, err := model.Fit(&train)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Finished fitting in %v.\n", time.Since(start))

	fmt.Println("Evaluating...")
	mrr, err := model.MRRScore(&test)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loss %v, MRR: %v\n", loss, mrr)

	expectedMrr := expectedMRR()
	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}

	fmt.Println("Predicting...")
	predictions, err := model.Predict([]int{1, 2, 3}, []int{100, 200, 300, 400})
	if err != nil {
		t.Errorf("Failed predictions %v", err)
	}
	if len(predictions) != 4 {
		t.Errorf("Got wrong number of predictions")
	}

	fmt.Println("Testing predict bounds checks...")
	predictions, err = model.Predict([]int{1, 2, 3}, []int{100, 200, 300, 400, 10000})
	if err == nil {
		t.Errorf("Should have errored with items out of range.")
	}

	fmt.Println("Serializing...")
	serialized, err := model.MarshalBinary()
	if err != nil {
		t.Errorf("Couldn't serialize %v", err)
	}

	fmt.Println("Deserializing...")
	deserializedModel := &ImplicitLSTMModel{}
	err = deserializedModel.UnmarshalBinary(serialized)
	if err != nil {
		t.Errorf("Couldn't deserialize")
	}

	fmt.Println("Evaluating deserialized model...")
	mrr, err = deserializedModel.MRRScore(&test)
	if err != nil {
		panic(err)
	}
	fmt.Printf("After deserialization: loss %v, MRR: %v\n", loss, mrr)

	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}

	fmt.Println("Testing model copies...")
	// Make a copy of the model, and free the model in the first model.
	// Make sure that using the model on the copy does not segfault, and
	// is handled correctly.
	var copy ImplicitLSTMModel = *model
	if copy.model == nil {
		t.Errorf("Copy model should be non-nil")
	}

	model.Free()
	if model.isTrained() {
		t.Errorf("Original model pointer should be nil.")
	}
	if copy.isTrained() {
		t.Errorf("Copy model pointer should be nil.")
	}
	mrr, err = copy.MRRScore(&test)
	if err == nil {
		t.Errorf("Freed copy shouldn't be able to score")
	}
}
