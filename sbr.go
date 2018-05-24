package sbr

//go:generate make build

/*
#cgo LDFLAGS: -L${SRCDIR}/sbr-sys/target/release -lsbr_sys
#include <sys/types.h>
#include <stdlib.h>
#include <sbr-sys/bindings.h>
*/
import "C"
import (
	"unsafe"
	"fmt"
	"math/rand"
)

type usize = C.size_t

type Interactions struct {
	numUsers   int
	numItems   int
	users      []usize
	items      []usize
	timestamps []usize
}

func NewInteractions(numUsers int, numItems int) Interactions {
	return Interactions{
		numUsers:   numUsers,
		numItems:   numItems,
		users:      make([]usize, 0),
		items:      make([]usize, 0),
		timestamps: make([]usize, 0),
	}
}

func (self *Interactions) Append(userId int, itemId int, timestamp int) error {
	if userId >= self.numUsers {
		self.numUsers = userId + 1
	}
	if itemId >= self.numItems {
		self.numItems = itemId + 1
	}

	self.users = append(self.users, usize(userId))
	self.items = append(self.items, usize(itemId))
	self.timestamps = append(self.timestamps, usize(timestamp))

	return nil
}

func (self *Interactions) NumItems() int {
	return self.numItems
}

func (self *Interactions) toFFI() (*C.InteractionsPointer, error) {
	result := C.interactions_new(C.size_t(self.numUsers),
		C.size_t(self.numItems),
		C.size_t(len(self.users)),
		&self.users[0],
		&self.items[0],
		&self.timestamps[0],
	)

	if result.error != nil {
		return nil, fmt.Errorf(C.GoString(result.error))
	}

	return result.value, nil
}

type ImplicitLSTMModel struct {
	NumItems          int
	MaxSequenceLength int
	ItemEmbeddingDim  int
	LearningRate      float32
	L2Penalty         float32
	NumThreads        int
	NumEpochs         int
	RandomSeed        [16]byte
	model             *C.ImplicitLSTMModelPointer
}

func NewImplicitLSTMModel(numItems int) *ImplicitLSTMModel {

	seed := make([]byte, 16)
	rand.Read(seed)

	var randomSeed [16]byte
	for idx := range randomSeed {
		randomSeed[idx] = seed[idx]
	}

	model := &ImplicitLSTMModel{
		NumItems:          numItems,
		MaxSequenceLength: 32,
		ItemEmbeddingDim:  32,
		LearningRate:      0.01,
		L2Penalty:         0.0,
		NumThreads:        1,
		NumEpochs:         10,
	}

	return model
}

func (self *ImplicitLSTMModel) Free() {
	if self.model != nil {
		C.implicit_lstm_free(self.model)
		self.model = nil
	}
}

func (self *ImplicitLSTMModel) Fit(data *Interactions) (float32, error) {
	if self.model == nil {

		var seed [16]C.uchar
		for idx, val := range self.RandomSeed {
			seed[idx] = C.uchar(val)
		}

		hyper := C.LSTMHyperparameters{
			num_items:           usize(self.NumItems),
			max_sequence_length: usize(self.MaxSequenceLength),
			item_embedding_dim:  usize(self.ItemEmbeddingDim),
			learning_rate:       C.float(self.LearningRate),
			l2_penalty:          C.float(self.L2Penalty),
			loss:                C.Hinge,
			optimizer:           C.Adagrad,
			num_threads:         usize(self.NumThreads),
			num_epochs:          usize(self.NumEpochs),
			random_seed:         seed,
		}
		result := C.implicit_lstm_new(hyper)

		if result.error != nil {
			return 0.0, fmt.Errorf(C.GoString(result.error))
		}

		self.model = result.value
	}

	dataFFI, err := data.toFFI()
	if err != nil {
		return 0.0, err
	}
	defer C.interactions_free(dataFFI)

	result := C.implicit_lstm_fit(self.model, dataFFI)

	if result.error != nil {
		return 0.0, fmt.Errorf(C.GoString(result.error))
	}

	return float32(*result.value), nil
}

func (self *ImplicitLSTMModel) Predict(interactionHistory []int, itemsToScore []int) ([]float32, error) {

	if self.model == nil {
		return nil, fmt.Errorf("Model has to be fit first.")
	}

	if len(interactionHistory) == 0 {
		return nil, fmt.Errorf("Interaction history must not be empty.")
	}

	if len(itemsToScore) == 0 {
		return nil, fmt.Errorf("Items to score must not be empty")
	}

	history := make([]usize, len(interactionHistory))
	items := make([]usize, len(itemsToScore))
	out := make([]C.float, len(itemsToScore))

	for i, v := range interactionHistory {
		if v >= self.NumItems {
			return nil, fmt.Errorf("Item ids must be smaller than NumItems")
		}
		history[i] = usize(v)
	}

	for i, v := range itemsToScore {
		if v >= self.NumItems {
			return nil, fmt.Errorf("Item ids must be smaller than NumItems")
		}
		items[i] = usize(v)
	}

	err := C.implicit_lstm_predict(self.model,
		&history[0],
		C.size_t(len(history)),
		&items[0],
		&out[0],
		C.size_t(len(out)))

	if err != nil {
		return nil, fmt.Errorf(C.GoString(err))
	}

	predictions := make([]float32, len(out))
	for i, v := range out {
		predictions[i] = float32(v)
	}

	return predictions, nil
}

func (self *ImplicitLSTMModel) MRRScore(data *Interactions) (float32, error) {
	if self.model == nil {
		return 0.0, fmt.Errorf("Model has to be fit first.")
	}

	dataFFI, err := data.toFFI()
	if err != nil {
		return 0.0, err
	}
	defer C.interactions_free(dataFFI)

	result := C.implicit_lstm_mrr_score(self.model, dataFFI)
	if result.error != nil {
		return 0.0, fmt.Errorf(C.GoString(result.error))
	}

	return float32(*result.value), nil
}

func (self *ImplicitLSTMModel) Serialize() ([]byte, error) {
	if self.model == nil {
		return nil, fmt.Errorf("Model has to be fit first.")
	}

	size := C.implicit_lstm_get_serialized_size(self.model)

	out := make([]byte, size)
	err := C.implicit_lstm_serialize(self.model,
		(*C.uchar)(unsafe.Pointer(&out[0])),
		usize(len(out)))

	if err != nil {
		return nil, fmt.Errorf(C.GoString(err))
	}

	return out, nil
}

func (self *ImplicitLSTMModel) Deserialize(data []byte) error {
	result := C.implicit_lstm_deserialize(
		(*C.uchar)(unsafe.Pointer(&data[0])),
		usize(len(data)))

	if result.error != nil {
		return fmt.Errorf(C.GoString(result.error))
	}

	self.model = result.value

	return nil
}
