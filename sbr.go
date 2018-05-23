package sbr

//go:generate make build

/*
#cgo LDFLAGS: libsbr_sys.so
#include <sys/types.h>
#include <stdlib.h>
#include <sbr-sys/bindings.h>
*/
import "C"
import (
	"fmt"
	"math/rand"
	"runtime"
)

type Interactions struct {
	numUsers   int
	numItems   int
	users      []C.int32_t
	items      []C.int32_t
	timestamps []C.int32_t
}

func NewInteractions(numUsers int, numItems int) Interactions {
	return Interactions{
		numUsers:   numUsers,
		numItems:   numItems,
		users:      make([]C.int32_t, 0),
		items:      make([]C.int32_t, 0),
		timestamps: make([]C.int32_t, 0),
	}
}

func (self *Interactions) Append(userId int, itemId int, timestamp int) error {
	if userId >= self.numUsers {
		self.numUsers = userId + 1
	}
	if itemId >= self.numItems {
		self.numItems = itemId + 1
	}

	self.users = append(self.users, C.int32_t(userId))
	self.items = append(self.items, C.int32_t(itemId))
	self.timestamps = append(self.timestamps, C.int32_t(timestamp))

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

	runtime.SetFinalizer(model, freeImplicitLSTMModel)

	return model
}

func freeImplicitLSTMModel(model *ImplicitLSTMModel) {
	if model.model != nil {
		fmt.Println("Freeing model")
		C.implicit_lstm_free(model.model)
	}
}

func (self *ImplicitLSTMModel) Fit(data *Interactions) (float32, error) {
	if self.model == nil {

		var seed [16]C.uchar
		for idx, val := range self.RandomSeed {
			seed[idx] = C.uchar(val)
		}

		hyper := C.LSTMHyperparameters{
			num_items:           C.uint64_t(self.NumItems),
			max_sequence_length: C.uint64_t(self.MaxSequenceLength),
			item_embedding_dim:  C.uint64_t(self.ItemEmbeddingDim),
			learning_rate:       C.float(self.LearningRate),
			l2_penalty:          C.float(self.L2Penalty),
			loss:                C.Hinge,
			optimizer:           C.Adagrad,
			num_threads:         C.uint64_t(self.NumThreads),
			num_epochs:          C.uint64_t(self.NumEpochs),
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
