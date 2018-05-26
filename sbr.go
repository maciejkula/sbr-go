// A recommender system package for Go.
//
// Sbr implements cutting-edge sequence-based recommenders: for every user, we examine what
// they have interacted up to now to predict what they are going to consume next.
//
// Installation requirements
//
// You will need the Rust compiler. You can install Rust from https://www.rust-lang.org/en-US/install.html by running
//  curl https://sh.rustup.rs -sSf | sh
//
// Installation
//
// Run
//  go get github.com/maciejkula/sbr-go
// followed by
//  make
// in the installation directory. This wil compile the package's native dependencies.
// The resulting libraries (in ./lib) must be available and in the dylib loading
// path at runtime.
//
// You may have to set
//  export CGO_LDFLAGS_ALLOW="-Wl.*"
// depending on your Go version.
package sbr

// Use both $ORIGIN and ${SRCDIR} for runtime lib loading. The first is relative to the
// executable location; the second uses an absolute link to the location where the binary
// was built.

/*
#cgo LDFLAGS: -L${SRCDIR}/lib -lsbr_sys -Wl,-rpath,\$ORIGIN/lib:${SRCDIR}/lib
#include <sys/types.h>
#include <stdlib.h>
#include <sbr-sys/bindings.h>
*/
import "C"
import (
	"fmt"
	"math/rand"
	"unsafe"
)

type Loss int
type Optimizer int

const (
	BPR     Loss      = 0
	Hinge   Loss      = 1
	Adam    Optimizer = 0
	Adagrad Optimizer = 1
)

// Helper for translating user and item ids into contiguous indices.
type Indexer struct {
	idToIdx map[string]int
	idxToId map[int]string
}

// Build a new indexer.
func NewIndexer() Indexer {
	return Indexer{
		idToIdx: make(map[string]int),
		idxToId: make(map[int]string),
	}
}

// Add a new id to the indexer, returning its model index.
func (self *Indexer) Add(id string) int {
	idx, ok := self.idToIdx[id]

	if !ok {
		idx = len(self.idToIdx)
		self.idToIdx[id] = idx
		return idx
	} else {
		return idx
	}
}

// Get the id from a model index.
func (self *Indexer) GetId(idx int) (string, bool) {
	id, ok := self.idxToId[idx]
	return id, ok
}

// Contains interactons for training the model.
type Interactions struct {
	numUsers   int
	numItems   int
	users      []C.size_t
	items      []C.size_t
	timestamps []C.size_t
}

// Construct new empty interactions.
func NewInteractions(numUsers int, numItems int) Interactions {
	return Interactions{
		numUsers:   numUsers,
		numItems:   numItems,
		users:      make([]C.size_t, 0),
		items:      make([]C.size_t, 0),
		timestamps: make([]C.size_t, 0),
	}
}

// Add a (user, item, timestamp) triple to the dataset.
func (self *Interactions) Append(userId int, itemId int, timestamp int) {
	if userId >= self.numUsers {
		self.numUsers = userId + 1
	}
	if itemId >= self.numItems {
		self.numItems = itemId + 1
	}

	self.users = append(self.users, C.size_t(userId))
	self.items = append(self.items, C.size_t(itemId))
	self.timestamps = append(self.timestamps, C.size_t(timestamp))
}

// Get the total number of distinct items in the data.
func (self *Interactions) NumItems() int {
	return self.numItems
}

// Get the total number of distinct users in the data.
func (self *Interactions) NumUsers() int {
	return self.numUsers
}

// Return the number of interactions.
func (self *Interactions) Len() int {
	return len(self.users)
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

// Split the interaction data into training and test sets. The data is split so that there is
// no overlap between users in training and test sets, making perfomance evaluation reflect
// the model's perfomance on entirely new users.
//
// Returns a tuple of (training, test) data.
func TrainTestSplit(data *Interactions, testFraction float64, rng *rand.Rand) (Interactions, Interactions) {

	testUsers := make(map[C.size_t]struct{})

	for idx := 0; idx < data.NumUsers(); idx++ {
		if rng.Float64() < testFraction {
			testUsers[C.size_t(idx)] = struct{}{}
		}
	}

	train := NewInteractions(data.NumUsers(), data.NumItems())
	test := NewInteractions(data.NumUsers(), data.NumItems())

	for i, uid := range data.users {
		_, ok := testUsers[uid]
		if ok {
			test.Append(int(uid), int(data.items[i]), int(data.timestamps[i]))
		} else {
			train.Append(int(uid), int(data.items[i]), int(data.timestamps[i]))
		}
	}

	return train, test
}

// An implicit-feedback LSTM-based sequence model.
type ImplicitLSTMModel struct {
	NumItems          int
	MaxSequenceLength int
	ItemEmbeddingDim  int
	LearningRate      float32
	L2Penalty         float32
	NumThreads        int
	NumEpochs         int
	Loss              Loss
	Optimizer         Optimizer
	RandomSeed        [16]byte
	// We use a double indirection scheme here to make
	// sure that if a copy of the model struct is created,
	// calling Free() on _any_ of the instances marks the
	// model as freed in _all_ the instances. Otherwise
	// we would have objects referencing a dead C pointer.
	model **C.ImplicitLSTMModelPointer
}

// Build a new model with a capacity to represent a certain number of items.
// In order to avoid leaking memory, the model must be freed usint its Free
// method once no longer in use.
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
		Loss:              Hinge,
		Optimizer:         Adagrad,
		LearningRate:      0.01,
		L2Penalty:         0.0,
		NumThreads:        1,
		NumEpochs:         10,
	}

	return model
}

func (self *ImplicitLSTMModel) isTrained() bool {
	return self.model != nil && *self.model != nil
}

// Free the memory associated with the underlying model.
//
// Unlike other methods of the model, calling Free is _not_
// thread safe. Use an external synchronisation method when
// freeing a model used from multiple goroutines.
func (self *ImplicitLSTMModel) Free() {
	if self.isTrained() {
		C.implicit_lstm_free(*self.model)
		*self.model = nil
		self.model = nil
	}
}

// Fit the model on the supplied data, returning the loss value after fitting.
// Calling this multiple times will resume training.
func (self *ImplicitLSTMModel) Fit(data *Interactions) (float32, error) {
	if self.model == nil {

		var seed [16]C.uchar
		for idx, val := range self.RandomSeed {
			seed[idx] = C.uchar(val)
		}

		var loss C.Loss
		var optimizer C.Optimizer

		if self.Optimizer == Adagrad {
			optimizer = C.Adagrad
		} else {
			optimizer = C.Adam
		}

		if self.Loss == BPR {
			loss = C.BPR
		} else {
			loss = C.Hinge
		}

		hyper := C.LSTMHyperparameters{
			num_items:           C.size_t(self.NumItems),
			max_sequence_length: C.size_t(self.MaxSequenceLength),
			item_embedding_dim:  C.size_t(self.ItemEmbeddingDim),
			learning_rate:       C.float(self.LearningRate),
			l2_penalty:          C.float(self.L2Penalty),
			loss:                loss,
			optimizer:           optimizer,
			num_threads:         C.size_t(self.NumThreads),
			num_epochs:          C.size_t(self.NumEpochs),
			random_seed:         seed,
		}
		result := C.implicit_lstm_new(hyper)

		if result.error != nil {
			return 0.0, fmt.Errorf(C.GoString(result.error))
		}

		self.model = &result.value
	}

	dataFFI, err := data.toFFI()
	if err != nil {
		return 0.0, err
	}
	defer C.interactions_free(dataFFI)

	result := C.implicit_lstm_fit(*self.model, dataFFI)

	if result.error != nil {
		return 0.0, fmt.Errorf(C.GoString(result.error))
	}

	return float32(*result.value), nil
}

// Make predictions. Provides scores for itemsToScore for a user who has
// seen interactionHistory items. Items in the history argument should be arranged
// chronologically, from the earliest seen item to the latest seen item.
//
// Returns a slice of scores for the supplied items, where a higher score indicates
// a better recommendation.
func (self *ImplicitLSTMModel) Predict(interactionHistory []int, itemsToScore []int) ([]float32, error) {

	if !self.isTrained() {
		return nil, fmt.Errorf("Model has to be fit first.")
	}

	if len(interactionHistory) == 0 {
		return nil, fmt.Errorf("Interaction history must not be empty.")
	}

	if len(itemsToScore) == 0 {
		return nil, fmt.Errorf("Items to score must not be empty")
	}

	history := make([]C.size_t, len(interactionHistory))
	items := make([]C.size_t, len(itemsToScore))
	out := make([]C.float, len(itemsToScore))

	for i, v := range interactionHistory {
		if v >= self.NumItems {
			return nil, fmt.Errorf("Item ids must be smaller than NumItems")
		}
		history[i] = C.size_t(v)
	}

	for i, v := range itemsToScore {
		if v >= self.NumItems {
			return nil, fmt.Errorf("Item ids must be smaller than NumItems")
		}
		items[i] = C.size_t(v)
	}

	err := C.implicit_lstm_predict(*self.model,
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

// Compute the mean reciprocal rank score of the model on supplied interaction data.
//
// Higher MRR values reflect better predictive performance of the model. The score
// is calculated by taking all but the last interactions of all users as their history,
// then making predictions for the last item they are going to see.
func (self *ImplicitLSTMModel) MRRScore(data *Interactions) (float32, error) {
	if !self.isTrained() {
		return 0.0, fmt.Errorf("Model has to be fit first.")
	}

	dataFFI, err := data.toFFI()
	if err != nil {
		return 0.0, err
	}
	defer C.interactions_free(dataFFI)

	result := C.implicit_lstm_mrr_score(*self.model, dataFFI)
	if result.error != nil {
		return 0.0, fmt.Errorf(C.GoString(result.error))
	}

	return float32(*result.value), nil
}

/// Serialize the underlying model into a byte array.
func (self *ImplicitLSTMModel) Serialize() ([]byte, error) {
	if !self.isTrained() {
		return nil, fmt.Errorf("Model has to be fit first.")
	}

	size := C.implicit_lstm_get_serialized_size(*self.model)

	out := make([]byte, size)
	err := C.implicit_lstm_serialize(*self.model,
		(*C.uchar)(unsafe.Pointer(&out[0])),
		C.size_t(len(out)))

	if err != nil {
		return nil, fmt.Errorf(C.GoString(err))
	}

	return out, nil
}

// Restore the model from a byte array.
func (self *ImplicitLSTMModel) Deserialize(data []byte) error {
	result := C.implicit_lstm_deserialize(
		(*C.uchar)(unsafe.Pointer(&data[0])),
		C.size_t(len(data)))

	if result.error != nil {
		return fmt.Errorf(C.GoString(result.error))
	}

	self.model = &result.value

	return nil
}
