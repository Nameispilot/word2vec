package main

import (
	"log"
	w "word2vec"
)

func main() {
	var err error
	var words []string
	var targets []float64

	//getting dataset from .csv
	if words, targets, err = w.MakeInputs("cmd/dataset.csv"); err != nil {
		log.Fatalf("%+v", err)
	}

	//feature hashing
	var arrOut []int
	arrOut, err = w.Hash(words)
	if err != nil {
		log.Fatalf("%+v", err)
	}

	inputWord := arrOut[0]
	outputs := arrOut[1:]

	err = w.Word2Vec(inputWord, outputs, targets)
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
