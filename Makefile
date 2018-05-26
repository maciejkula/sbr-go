ifeq ($(OS),Windows_NT)
	FEATURES += ""
	CP = "xcopy .\sbr-sys\target\release\*dll .\lib /sy"
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		FEATURES += "mkl"
		CP = 'find ./sbr-sys/target/release -name "*.so" -type f -exec cp {} ./lib \;'
	endif
	ifeq ($(UNAME_S),Darwin)
		FEATURES += "mkl"
		CP = 'find ./sbr-sys/target/release -name "*.dylib" -type f -exec cp {} ./lib \;'
	endif
endif

all:
	git submodule init
	git submodule update
	cd sbr-sys; RUSTFLAGS="-C target-cpu=native" cargo build --features=$(FEATURES) --release
	mkdir -p ./lib
	@eval $(CP)

