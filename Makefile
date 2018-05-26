ifeq ($(OS),Windows_NT)
	FEATURES += ""
	CP = 'mkdir lib; xcopy /s/y "sbr-sys\target\release\*.dll" "lib\"'
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		FEATURES += "mkl"
		CP = 'mkdir -p ./lib; find ./sbr-sys/target/release -name "*.so" -type f -exec cp {} ./lib \;'
	endif
	ifeq ($(UNAME_S),Darwin)
		FEATURES += "mkl"
		CP = 'mkdir -p ./lib; find ./sbr-sys/target/release -name "*.dylib" -type f -exec cp {} ./lib \;'
	endif
endif

all:
	git submodule init
	git submodule update
	cd sbr-sys; RUSTFLAGS="-C target-cpu=native" cargo build --features=$(FEATURES) --release
	@eval $(CP)

