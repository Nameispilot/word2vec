package word2vec

import (
	"encoding/csv"
	"os"
	"strconv"

	"github.com/pkg/errors"
)

func MakeInputs(fileName string) ([]string, []float64, error) {

	// Open the .csv file
	file, err := os.Open(fileName)

	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = 3

	// Read all the data from the file
	rawData, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	//var inputs string
	outputs := make([]string, len(rawData)+1)
	targets := make([]float64, len(rawData))

	//var inputsIdx int
	var outputsIdx int
	var targetsIdx int

	var fl = 0

	for _, record := range rawData {
		for i, value := range record {

			if i == 0 {
				if fl == 0 {
					outputs[outputsIdx] = value
					outputsIdx++
					fl = 1
					continue
				}
			}
			if i == 1 {
				outputs[outputsIdx] = value
				outputsIdx++
				continue
			}
			if i == 2 {
				targets[targetsIdx], err = strconv.ParseFloat(value, 64)
				if err != nil {
					return nil, nil, errors.Wrap(err, "Error while parsing!")
				}
				targetsIdx++
			}
		}
	}
	return outputs, targets, nil
}
