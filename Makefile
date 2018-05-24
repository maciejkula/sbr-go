ifeq ($(OS),Windows_NT)
	FEATURES += ""
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		FEATURES += "mkl"
	endif
	ifeq ($(UNAME_S),Darwin)
		FEATURES += "mkl"
	endif
endif

all:
	git submodule init
	git submodule update
	cd sbr-sys; RUSTFLAGS="-C target-cpu=native" cargo build --features=$(FEATURES) --release
