package sbr

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"testing"
)

func TestMovielens100K(t *testing.T) {
	data, err := GetMovielens()
	if err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(42))
	train, test := TrainTestSplit(data, 0.2, rng)

	fmt.Printf("Train len %v, test len %v\n", train.Len(), test.Len())

	model := NewImplicitLSTMModel(data.NumItems())

	// Set the hyperparameters.
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

	loss, err := model.Fit(&train)
	if err != nil {
		panic(err)
	}

	mrr, err := model.MRRScore(&test)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loss %v, MRR: %v\n", loss, mrr)

	expectedMrr := float32(0.07)
	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}

	predictions, err := model.Predict([]int{1, 2, 3}, []int{100, 200, 300, 400})
	if err != nil {
		t.Errorf("Failed predictions %v", err)
	}
	if len(predictions) != 4 {
		t.Errorf("Got wrong number of predictions")
	}

	predictions, err = model.Predict([]int{1, 2, 3}, []int{100, 200, 300, 400, 10000})
	if err == nil {
		t.Errorf("Should have errored with items out of range.")
	}

	serialized, err := model.Serialize()
	if err != nil {
		t.Errorf("Couldn't serialize %v", err)
	}

	deserializedModel := &ImplicitLSTMModel{}
	modelJson, _ := json.Marshal(model)
	_ = json.Unmarshal(modelJson, deserializedModel)

	err = deserializedModel.Deserialize(serialized)
	if err != nil {
		t.Errorf("Couldn't deserialize")
	}

	mrr, err = deserializedModel.MRRScore(&test)
	if err != nil {
		panic(err)
	}
	fmt.Printf("After deserialization: loss %v, MRR: %v\n", loss, mrr)

	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}
}
