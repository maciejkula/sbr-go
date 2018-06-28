package main

import (
	"archive/zip"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"runtime"
	"strconv"

	"github.com/intel-go/cpuid"
	"github.com/schollz/progressbar"
)

const BASE_URL = "https://github.com/maciejkula/sbr-sys/releases/download/v0.4.0/"

const SSE = "sse"
const AVX = "avx"

func getSIMDCapability() string {
	if cpuid.HasExtendedFeature(cpuid.AVX2) {
		return AVX
	}
	if cpuid.HasFeature(cpuid.AVX) {
		return AVX
	}

	return SSE
}

func download() error {
	tempDir, err := ioutil.TempDir("", "sbr-go")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	tempFileName := "sbd_dist.zip"
	tempFilePath := path.Join(tempDir, tempFileName)

	capability := getSIMDCapability()

	url := BASE_URL
	var archiveFilename string

	if runtime.GOOS == "linux" {
		url += fmt.Sprintf("linux_%v_libsbr_sys.a.zip", capability)
		archiveFilename = "libsbr_sys.a"
	} else if runtime.GOOS == "darwin" {
		url += fmt.Sprintf("darwin_%v_libsbr_sys.a.zip", capability)
		archiveFilename = "libsbr_sys.a"
	} else {
		return fmt.Errorf("Unsupported OS: %v", runtime.GOOS)
	}

	// Get the data
	fmt.Println("Downloading binary distribution...")
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Create the file
	out, err := os.Create(tempFilePath)
	if err != nil {
		return err
	}

	// Writer the body to file
	size, err := strconv.Atoi(resp.Header.Get("Content-Length"))
	if err != nil {
		return err
	}
	chunkSize := 256 * 1024
	bar := progressbar.New(size / chunkSize)

	for err != io.EOF {
		_, err = io.CopyN(out, resp.Body, int64(chunkSize))
		if err != nil && err != io.EOF {
			return err
		}
		bar.Add(1)
	}
	out.Close()

	archive, err := zip.OpenReader(tempFilePath)
	defer archive.Close()

	// Create the file
	err = os.MkdirAll("lib", os.ModePerm)
	if err != nil {
		return err
	}

	fmt.Println("Unpacking archive...")
	destinationPath := path.Join("lib", path.Base(archiveFilename))
	destination, err := os.Create(destinationPath)
	if err != nil {
		return err
	}
	defer destination.Close()

	fileNames := make([]string, 0)

	for _, file := range archive.File {

		fileNames = append(fileNames, file.FileHeader.Name)

		if file.FileHeader.Name == archiveFilename {
			archiveFile, err := file.Open()
			if err != nil {
				return err
			}

			_, err = io.Copy(destination, archiveFile)
			if err != nil {
				return err
			}

			fmt.Println("Done.")
			defer archiveFile.Close()
			return nil
		}
	}

	return fmt.Errorf("Release binary not found in downloaded archive: %v not in %v",
		archiveFilename, fileNames)
}

func main() {
	err := download()
	if err != nil {
		log.Fatal(err)
	}
}
