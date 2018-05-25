package sbr

import (
	"bufio"
	"encoding/csv"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
)

const dataPath = "/.sbr/"

func userHomeDir() string {
	if runtime.GOOS == "windows" {
		home := os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
		if home == "" {
			home = os.Getenv("USERPROFILE")
		}
		return home
	}
	return os.Getenv("HOME")
}

func readData(path string) (*Interactions, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	interactions := NewInteractions(100, 100)
	reader := bufio.NewReader(file)
	_, _, err = reader.ReadLine() // Skip the header
	if err != nil {
		return nil, err
	}
	csvReader := csv.NewReader(reader)

	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		userId, err := strconv.ParseInt(record[0], 10, 32)
		if err != nil {
			return nil, err
		}
		itemId, err := strconv.ParseInt(record[1], 10, 32)
		if err != nil {
			return nil, err
		}
		timestamp, err := strconv.ParseInt(record[3], 10, 32)
		if err != nil {
			return nil, err
		}

		interactions.Append(int(userId),
			int(itemId),
			int(timestamp))
	}

	return &interactions, nil
}

// Download and return the Movielens 100K dataset.
func GetMovielens() (*Interactions, error) {
	dataDir := filepath.Join(userHomeDir(), dataPath)

	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		err := os.MkdirAll(dataDir, os.ModePerm)
		if err != nil {
			return nil, err
		}
	}

	dataPath := filepath.Join(dataDir, "movielens.csv")

	if _, err := os.Stat(dataPath); os.IsNotExist(err) {
		out, err := os.Create(dataPath)
		if err != nil {
			return nil, err
		}

		resp, err := http.Get("https://github.com/maciejkula/sbr-rs/raw/master/data.csv")
		if err != nil {
			return nil, err
		}

		_, err = io.Copy(out, resp.Body)
		if err != nil {
			return nil, err
		}
		err = out.Close()
		if err != nil {
			return nil, err
		}
		err = resp.Body.Close()
		if err != nil {
			return nil, err
		}
	}

	return readData(dataPath)
}
