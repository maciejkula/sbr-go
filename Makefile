build:
	git submodule init
	git submodule update
	cd sbr-sys; cargo build --release
	cp sbr-sys/target/release/libsbr_sys.* .
