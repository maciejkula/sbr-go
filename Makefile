ifeq ($(OS),Windows_NT)
	FEATURES += ""
	EXT = "*.dll"
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		FEATURES += "mkl"
		EXT = "*.so"
	endif
	ifeq ($(UNAME_S),Darwin)
		FEATURES += "mkl"
		EXT = "*.dylib"
	endif
endif

# Use pre-built binaries.
all:
	cd build && go build && ./build && cp -r lib ../

# Build everything from source.
source:
	git submodule init
	git submodule update
	cd sbr-sys; RUSTFLAGS="-C target-cpu=native" cargo build --features=$(FEATURES) --release
	mkdir -p ./lib; find ./sbr-sys/target/release -name "$(EXT)" -type f -exec cp {} ./lib \;
