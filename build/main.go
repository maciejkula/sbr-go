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
)

const BASE_URL = "https://github.com/maciejkula/sbr-sys/releases/download/"
const LINUX_URL = "untagged-8b9d185393b92ca20ccb/libsbr_linux.zip"
const DARWIN_URL = "untagged-8438cacd506366a30457/libsbr_darwin.zip"

func download() error {
	tempDir, err := ioutil.TempDir("", "sbr-go")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	tempFileName := "sbd_dist.zip"
	tempFilePath := path.Join(tempDir, tempFileName)

	url := BASE_URL
	var archiveFilename string

	if runtime.GOOS == "linux" {
		url += LINUX_URL
		archiveFilename = "linux/sse/libsbr_sys.a"
	} else if runtime.GOOS == "darwin" {
		url += DARWIN_URL
		archiveFilename = "darwin/sse/libsbr_sys.a"
	} else {
		return fmt.Errorf("Unsupported OS: %v", runtime.GOOS)
	}

	// Get the data
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
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return err
	}
	out.Close()

	archive, err := zip.OpenReader(tempFilePath)
	defer archive.Close()

	// Create the file
	err = os.MkdirAll("lib", os.ModePerm)
	if err != nil {
		return err
	}

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

			defer archiveFile.Close()
		}
	}

	return fmt.Errorf("Release binary not found in downloaded archive: %v not in %v",
		archiveFilename, fileNames)
}

func main() {
	download()
}
