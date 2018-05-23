package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	sbr "github.com/maciejkula/sbr-go"
)

func readData(path string) (*sbr.Interactions, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	interactions := sbr.NewInteractions(100, 100)
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

func main() {
	data, err := readData("data.csv")
	if err != nil {
		panic(err)
	}

	model := sbr.NewImplicitLSTMModel(data.NumItems())

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
}
