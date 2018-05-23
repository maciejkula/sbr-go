build:
	git submodule init
	git submodule update
	cd sbr-sys; RUSTFLAGS="-C target-cpu=native" cargo build --release
