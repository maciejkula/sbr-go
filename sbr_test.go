package sbr

import (
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func runTest(t *testing.T, model *ImplicitLSTMModel, train *Interactions,
	test *Interactions, expectedMrr float32) {

	t.Log("Fitting...")
	start := time.Now()
	loss, err := model.Fit(train)
	if err != nil {
		panic(err)
	}
	t.Logf("Finished fitting in %v.\n", time.Since(start))

	t.Log("Evaluating...")
	mrr, err := model.MRRScore(test)
	if err != nil {
		panic(err)
	}
	t.Logf("Loss %v, MRR: %v\n", loss, mrr)

	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}

	t.Log("Predicting...")
	predictions, err := model.Predict([]int{1, 2, 3}, []int{100, 200, 300, 400})
	if err != nil {
		t.Errorf("Failed predictions %v", err)
	}
	if len(predictions) != 4 {
		t.Errorf("Got wrong number of predictions")
	}

	t.Log("Testing predict bounds checks...")
	predictions, err = model.Predict([]int{1, 2, 3}, []int{100, 200, 300, 400, 10000})
	if err == nil {
		t.Errorf("Should have errored with items out of range.")
	}

	t.Log("Serializing...")
	serialized, err := model.MarshalBinary()
	if err != nil {
		t.Errorf("Couldn't serialize %v", err)
	}

	t.Log("Deserializing...")
	deserializedModel := &ImplicitLSTMModel{}
	err = deserializedModel.UnmarshalBinary(serialized)
	if err != nil {
		t.Errorf("Couldn't deserialize")
	}

	t.Log("Evaluating deserialized model...")
	mrr, err = deserializedModel.MRRScore(test)
	if err != nil {
		panic(err)
	}
	t.Logf("After deserialization: loss %v, MRR: %v\n", loss, mrr)

	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}

	t.Log("Testing model copies...")
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
	mrr, err = copy.MRRScore(test)
	if err == nil {
		t.Errorf("Freed copy shouldn't be able to score")
	}
}

func TestHinge(t *testing.T) {
	data, err := GetMovielens()
	if err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(42))
	train, test := TrainTestSplit(data, 0.2, rng)

	t.Logf("Train len %v, test len %v\n", train.Len(), test.Len())

	t.Log("Building model...")
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

	var expectedMRR float32

	if runtime.GOOS == "linux" {
		// OpenBLAS build.
		expectedMRR = 0.07
	} else {
		// Accelerate build.
		expectedMRR = 0.068
	}

	runTest(t, model, &train, &test, expectedMRR)
}

func TestWARP(t *testing.T) {
	data, err := GetMovielens()
	if err != nil {
		panic(err)
	}

	rng := rand.New(rand.NewSource(42))
	train, test := TrainTestSplit(data, 0.2, rng)

	t.Logf("Train len %v, test len %v\n", train.Len(), test.Len())

	t.Log("Building model...")
	model := NewImplicitLSTMModel(data.NumItems())

	// Set the hyperparameters.
	model.MaxSequenceLength = 32
	model.ItemEmbeddingDim = 32
	model.LearningRate = 0.26
	model.L2Penalty = 0.004
	model.NumEpochs = 15
	model.NumThreads = 1
	model.Loss = WARP
	model.Optimizer = Adagrad

	// Set random seed
	var randomSeed [16]byte
	for idx := range randomSeed {
		randomSeed[idx] = 42
	}
	model.RandomSeed = randomSeed

	var expectedMRR float32

	if runtime.GOOS == "linux" {
		// OpenBLAS build.
		expectedMRR = 0.07
	} else {
		// Accelerate build.
		expectedMRR = 0.09
	}

	runTest(t, model, &train, &test, expectedMRR)
}

func TestDeallocation(t *testing.T) {
	numItems := 1000000
	data := NewInteractions(1, numItems)

	for i := 0; i < 1000; i++ {
		data.Append(0, i, i)
	}

	for i := 0; i < 100; i++ {
		model := NewImplicitLSTMModel(numItems)
		_, err := model.Fit(&data)
		if err != nil {
			t.Errorf("Can't fit: %v", err)
		}

		runtime.GC()
	}
}
