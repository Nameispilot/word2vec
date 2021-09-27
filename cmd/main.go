package main

import (
	"fmt"
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
	arrOut, hashTable := w.Hash(words)

	var inputWord int
	for i := range hashTable {
		if hashTable[i] == words[0] {
			inputWord = i
			break
		}
	}

	outputs := arrOut[1:]

	err = w.Word2Vec(inputWord, outputs, targets)
	if err != nil {
		fmt.Println(err)
	}
}
