package sbr

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"testing"
)

func readData(path string) (*Interactions, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	interactions := NewInteractions(100, 100)
	reader := bufio.NewReader(file)
	reader.ReadLine() // Skip the header
	csvReader := csv.NewReader(reader)

	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		userId, err := strconv.ParseInt(record[0], 10, 32)
		if err != nil {
			return nil, err
		}
		itemId, err := strconv.ParseInt(record[1], 10, 32)
		if err != nil {
			return nil, err
		}
		timestamp, err := strconv.ParseInt(record[3], 10, 32)
		if err != nil {
			return nil, err
		}

		err = interactions.Append(int(userId),
			int(itemId),
			int(timestamp))
		if err != nil {
			return nil, err
		}
	}

	return &interactions, nil
}

func TestMovielens100K(t *testing.T) {
	data, err := readData("data.csv")
	if err != nil {
		panic(err)
	}

	model := NewImplicitLSTMModel(data.NumItems())

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

	loss, err := model.Fit(data)
	if err != nil {
		panic(err)
	}

	mrr, err := model.MRRScore(data)
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

	mrr, err = deserializedModel.MRRScore(data)
	if err != nil {
		panic(err)
	}
	fmt.Printf("After deserialization: loss %v, MRR: %v\n", loss, mrr)

	if mrr < expectedMrr {
		t.Errorf("MRR smaller than %v", expectedMrr)
	}
}
